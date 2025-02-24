from fast_graphrag import GraphRAG
from rdflib import Graph, URIRef, Literal, Dataset
from rdflib.namespace import RDF, OWL, RDFS, SKOS, Namespace, DefinedNamespace
import networkx as nx
import os
import logging
import argparse
from typing import Dict, Optional
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for ontology data
prefixes: Dict[str, str] = {}
domain: str = ""
entity_types: list[str] = []
ontology: Graph = Graph()
curie_map: Dict[str, URIRef] = {}  # Case-insensitive CURIE lookup

# System namespace for GraphRAG
SE = Namespace("https://models.data.world/graphrag/")  # type: ignore


def load_ontology(file_path: str) -> None:
    """
    Load ontology from TTL file and set global variables.
    """
    global prefixes, domain, entity_types, ontology, curie_map
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ontology file not found: {file_path}")
    
    # Read the raw content
    with open(file_path) as f:
        domain = f.read()
    
    # Parse with rdflib to extract prefixes and classes
    try:
        ontology.parse(data=domain, format='turtle')
    except Exception as e:
        logger.error(f"Failed to parse ontology file: {e}")
        raise ValueError(f"Invalid ontology file: {e}")
    
    prefixes = dict(ontology.namespaces())
    
    # Get all classes and their labels
    entity_types = []
    logger.info("Raw entity types from ontology:")
    for class_uri in ontology.subjects(RDF.type, OWL.Class):
        for label in ontology.objects(class_uri, RDFS.label):
            logger.info(f"Class URI: {class_uri}, Label: {label}")
            entity_types.append(str(label))

    # Build case-insensitive CURIE map
    for s, p, o in ontology:
        if isinstance(s, URIRef):
            for prefix, uri in prefixes.items():
                if str(s).startswith(str(uri)):
                    local = str(s)[len(str(uri)):]
                    curie = f"{prefix}:{local}"
                    curie_map[curie.lower()] = s

def make_curie_resolver(prefixes: Dict[str, str], curie_map: Dict[str, URIRef]):
    """Create a closure that captures the prefix and curie mappings"""
    def resolve_curie(curie: str) -> Optional[URIRef]:
        """Resolve a CURIE using case-insensitive lookup."""
        print(f"Resolving CURIE: {curie}")
        if not curie or ':' not in curie:
            return None
        
        # Try case-insensitive lookup first
        if curie.lower() in curie_map:
            print(f"Resolved to: {curie_map[curie.lower()]}")
            return curie_map[curie.lower()]
        
        # Fall back to direct resolution if not found
        prefix, local = curie.split(':', 1)
        if prefix.lower() in [p.lower() for p in prefixes]:
            prefix = next(p for p in prefixes if p.lower() == prefix.lower())
            print(f"Resolved to: {prefixes[prefix] + local}")
            return URIRef(prefixes[prefix] + local)
        return None
        
    return resolve_curie

def safe_uri_name(name: str) -> str:
    """Convert a string to a safe URI component by replacing spaces with underscores"""
    return name.replace(" ", "_")

def apply_hygiene_rules(output_graph: Graph, ontology_graph: Graph) -> Graph:
    """Apply all hygiene rules from the hygiene directory to fix the output graph."""
    # Create dataset with both graphs
    ds = Dataset()
    output_uri = URIRef("https://models.data.world/graphrag/output")
    ontology_uri = URIRef("https://models.data.world/graphrag/ontology")
    
    # Get/create the graphs and add the triples
    output = ds.graph(output_uri)
    ontology = ds.graph(ontology_uri)
    
    # Copy namespace bindings
    for prefix, namespace in output_graph.namespaces():
        output.bind(prefix, namespace)
    for prefix, namespace in ontology_graph.namespaces():
        ontology.bind(prefix, namespace)
    
    # Add all triples from our input graphs
    for triple in output_graph:
        output.add(triple)
    for triple in ontology_graph:
        ontology.add(triple)
    
    # Find and apply all hygiene rules
    hygiene_dir = "./hygiene"
    if os.path.exists(hygiene_dir):
        for filename in os.listdir(hygiene_dir):
            if filename.endswith('.rq'):
                rule_path = os.path.join(hygiene_dir, filename)
                logger.info(f"Applying hygiene rule from {filename}")
                with open(rule_path) as f:
                    update_query = f.read()
                    ds.update(update_query)
    
    # Get the updated output graph and copy its namespace bindings
    result = Graph()
    output = ds.graph(output_uri)
    for prefix, namespace in output.namespaces():
        result.bind(prefix, namespace)
    for triple in output:
        result.add(triple)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Process ontology and text for GraphRAG')
    parser.add_argument('-o', '--ontology', help='Path to ontology file (TTL format)',
                       default='./ontology.ttl')
    parser.add_argument('--renew', action='store_true', 
                       help='Delete existing graph files and force regeneration')
    args = parser.parse_args()
    
    # If renew flag is set, clean up old files
    if args.renew:
        logger.info("Cleaning up old graph files...")
        working_dir = "./print3D_example"
        if os.path.exists(working_dir):
            for file in os.listdir(working_dir):
                file_path = os.path.join(working_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f'Error deleting {file_path}: {e}')
            logger.info("Cleanup complete")

    # Load ontology
    try:
        load_ontology(args.ontology)
        logger.info(f"Loaded ontology from {args.ontology}")
        logger.info(f"Found prefixes: {prefixes}")
        logger.info(f"Found entity types: {entity_types}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(e)
        return

    # Create resolver with context
    resolver = make_curie_resolver(prefixes, curie_map)
    
    # Create GraphRAG instance
    grag = GraphRAG(
        working_dir="./print3D_example",
        domain=domain,
        example_queries="\n".join([
            "How does one printer compare to another?",
            "What are the different types of printers?",
            "What are the different types of materials?",
            "What are the different types of nozzles?",
            "What nozzle is best for a particular material?"
        ]),
        entity_types=entity_types,
        config=GraphRAG.Config(
            resolve_curie=resolver  # Pass the resolver instead of the maps
        )
    )

    # Process content and create graph
    with open("./printerreview.txt") as f:
        content = f.read()
        logger.info(f"Read {len(content)} characters from printerreview.txt")
        logger.info("Starting content insertion...")
        grag.insert(content)
        logger.info("Completed content insertion")

    # Save and load the graph
    graphml_path = os.path.join(grag.working_dir, "graph.graphml")
    print(f"Looking for graph at: {graphml_path}")
    grag.save_graphml(graphml_path)
    internal_graph = nx.read_graphml(graphml_path)
    
    # Debug the loaded graph
    logger.info("\nLoaded GraphML edges:")
    for s, t, data in internal_graph.edges(data=True):
        logger.info(f"Edge {s}->{t}: {data.keys()}")  # See what attributes are preserved

    # Create RDF graph
    g = Graph()
    g.bind("gr", SE)  # 'gr' for GraphRAG
    
    # Copy prefixes from ontology
    for prefix, uri in prefixes.items():
        g.bind(prefix, uri)

    # First define our classes as OWL classes
    for entity_type in entity_types:
        class_uri = SE[safe_uri_name(entity_type)]
        g.add((class_uri, RDF.type, OWL.Class))

    # Process nodes
    node_uris = {}  # Keep track of URIs for relationship processing
    print("\nNodes in graph:")
    for node, data in internal_graph.nodes(data=True):
        print(f"Node: {node}")
        print(f"Data: {data}")
        node_type = data.get('type')
        name = data.get('name', node)
        instance_uri = SE[safe_uri_name(name)]
        node_uris[node] = instance_uri
        
        # Try to resolve type as CURIE first
        if node_type and ':' in node_type:
            resolved_type = resolver(node_type)
            if resolved_type is not None:
                class_uri = resolved_type
            else:
                class_uri = SE.UNKNOWN
        # Then try matching against entity type labels
        elif node_type and node_type.lower() in [et.lower() for et in entity_types]:
            class_uri = SE[safe_uri_name(next(et for et in entity_types if et.lower() == node_type.lower()))]
        else:
            class_uri = SE.UNKNOWN
            
        if class_uri == SE.UNKNOWN:
            g.add((class_uri, RDF.type, OWL.Class))  # Define UNKNOWN as a class if not already done
            
        print(f"Adding triple: {instance_uri} rdf:type {class_uri}")
        g.add((instance_uri, RDF.type, class_uri))
        
        if 'name' in data:
            g.add((instance_uri, RDFS.label, Literal(data['name'])))
        
        if 'description' in data:
            g.add((instance_uri, SKOS.definition, Literal(data['description'])))

    # Process relationships
    print("\nRelationships in graph:")
    for source, target, edge_data in internal_graph.edges(data=True):
        print(f"Edge: {source} -> {target}")
        print(f"Edge Data: {edge_data}")
        
        if source in node_uris and target in node_uris:
            source_uri = node_uris[source]
            target_uri = node_uris[target]
            
            # Default predicate
            predicate = SE.relatedTo
            
            logger.info(f"\nProcessing edge property: {edge_data.get('property', 'no property')}")
            if 'property' in edge_data and edge_data['property']:
                logger.info("About to resolve CURIE...")
                resolved = resolver(edge_data['property'])
                if resolved is not None:
                    predicate = resolved
                    # Check if this is a new property
                    if not any(ontology.triples((resolved, RDF.type, None))):
                        g.add((resolved, RDF.type, SE.NewProperty))
            
            print(f"Adding triple: {source_uri} {predicate} {target_uri}")
            g.add((source_uri, predicate, target_uri))
            
            if 'description' in edge_data:
                print(f"Adding relationship description: {edge_data['description']}")
                g.add((source_uri, SKOS.note, Literal(edge_data['description'])))

    # Apply hygiene rules and save RDF graph
    g = apply_hygiene_rules(g, ontology)
    g.serialize(destination="print3D_knowledge.ttl", format="turtle")
    print("\nSaved RDF to print3D_knowledge.ttl")

if __name__ == "__main__":
    main()
