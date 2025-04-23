from collections import defaultdict
from io import BytesIO
import base64
from fast_graphrag import GraphRAG
from fast_graphrag._utils import get_event_loop
from rdflib import Graph, URIRef, Literal, Dataset
from rdflib.namespace import RDF, OWL, RDFS, SKOS, Namespace, DefinedNamespace, XSD
import networkx as nx
import os
import logging
import argparse
from typing import Dict, Optional
import shutil
import requests
from urllib.parse import urlparse



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

def get_local_part(url):
    parsed_url = urlparse(url)
    local_part = parsed_url.fragment if parsed_url.fragment else parsed_url.path.split('/')[-1]
    return local_part

def load_ontology(ontology_data: str) -> None:
    """
    Load ontology from TTL string and set global variables.
    """
    global prefixes, domain, entity_types, ontology, curie_map

    print (ontology_data)
    
    # ontology_data is the direct input now, so no need to read files
    domain = ontology_data 
    temp: Graph = Graph ()
    # Parse with rdflib to extract prefixes and classes
    try:
        temp.parse(data=domain, format='turtle')
    except Exception as e:
        logger.error(f"Failed to parse ontology file: {e}")
        raise ValueError(f"Invalid ontology file: {e}")


    # Collect declared classes
    good = set()
    for s, p, o in temp.triples((None, RDF.type, OWL.Class)):
        if get_local_part(s) != "UNKNOWN":
            good.add(s)
    for s, p, o in temp.triples((None, RDF.type, OWL.ObjectProperty)):
        good.add(s)
    for s, p, o in temp.triples((None, RDF.type, OWL.DatatypeProperty)):
        good.add(s)
    for s, p, o in temp.triples((None, RDF.type, OWL.Ontology)):
        good.add(s)


    for s, p, o in temp:
        if s in good:
            ontology.add((s, p, o))


    for prefix, uri in temp.namespace_manager.namespaces():
        ontology.namespace_manager.bind(prefix, uri)            
    
    ontology.serialize(destination='intermediate_ontology.ttl', format='turtle')
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
            for label in ontology.objects(s, RDFS.label):
                curie_map[label.lower()] = s
                print (f"Adding <{s}> as URI for key '{label.lower()}'")
            curie_map[get_local_part(s).lower()] = s
            print (f"Adding <{s}> as URI for key '{get_local_part(s).lower()}'")
            for prefix, uri in prefixes.items():
                if str(s).startswith(str(uri)):
                    local = str(s)[len(str(uri)):]
                    curie = f"{prefix}:{local}"
                    curie_map[curie.lower()] = s
                    print (f"Adding <{s}> as URI for key '{curie.lower()}'")

                    

                        

                    
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
            return URIRef(prefixes[prefix] + safe_uri_name(local))
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
    ont = ds.graph(ontology_uri)
    for prefix, namespace in output.namespaces():
        result.bind(prefix, namespace)
    for triple in output:
        result.add(triple)
    for triple in ont:
        result.add(triple)
    
    return result


def main():
    global ontology
    parser = argparse.ArgumentParser(description='Process ontology and text for GraphRAG')
    parser.add_argument('-v', '--version', help='Download specific version from data.world',
                       default=None)

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
        # Download data from the data.world API
        if args.version:
            api_token = os.getenv('DW_AUTH_TOKEN')
            headers = {"Authorization": f"Bearer {api_token}"}
            url = f"https://api.data.world/v0/file_download/{args.version}/ontology.ttl"
            response = requests.get(url, headers=headers)
            if not response.ok:
                response.raise_for_status()
            ontology_data = response.text
            load_ontology(ontology_data)
        else:
            # If no version specified, read from local ontology.ttl file
            with open('ontology.ttl', 'r') as file:
                ontology_data = file.read()
                load_ontology(ontology_data)
        logger.info("Loaded ontology")
        logger.info(f"Found prefixes: {prefixes}")
        logger.info(f"Found entity types: {entity_types}")
    except (FileNotFoundError, ValueError, requests.HTTPError) as e:
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
    with open("./input.txt") as f:
        content = f.read()
        logger.info(f"Read {len(content)} characters from input.txt")
        logger.info("Starting content insertion...")
        grag.insert(content)
        logger.info("Completed content insertion")

    # Instead of using GraphML, directly access the GraphRAG model
    # Get node and edge information from the state manager
    graph_storage = grag.state_manager.graph_storage
    chunk_storage = grag.state_manager.chunk_storage
    
    # Create RDF graph
    g = Graph()
    g.bind("gr", SE)  # 'gr' for GraphRAG
    
    # Copy prefixes from ontology
    for prefix, uri in prefixes.items():
        g.bind(prefix, uri)

    # Need to run in event loop to get the async data
    async def build_rdf_graph():
        try:
            # Process nodes from the graph storage
            node_count = await graph_storage.node_count()
            print(f"\nTotal nodes in graph: {node_count}")
            
            # Track node URIs for building relationships
            node_uris = {}
            node_names = {}
            
            # Process all nodes
            for i in range(node_count):
                node = await graph_storage.get_node_by_index(i)
                if node is None:
                    continue
                    
                name = node.name
                node_type = node.type
                description = node.description
                
                print(f"Node: {name}")
                print(f"Type: {node_type}")
                print(f"Description: {description}")
                
                # Create URI for this entity
                instance_uri = SE[safe_uri_name(name)]
                node_uris[name] = instance_uri
                node_names[i] = name
                
                # Try to resolve type as CURIE first
                if node_type and ':' in node_type:
                    resolved_type = resolver(node_type)
                    if resolved_type is not None:
                        class_uri = resolved_type
                    else:
                        class_uri = SE.UNKNOWN
                # Then try matching against entity type labels
                elif node_type and node_type.lower() in [et.lower() for et in entity_types]:
                    class_uri = curie_map.get(node_type.lower(), None)
                    if class_uri is None:
                        print(f"\n\nNode type {node_type} resulted in None for CURIE resolution\n\n")
                        class_uri = SE.UNKNOWN
                else:
                    class_uri = SE.UNKNOWN
                    
                if class_uri == SE.UNKNOWN:
                    g.add((class_uri, RDF.type, OWL.Class))  # Define UNKNOWN as a class if not already done
                    
                print(f"Adding triple: {instance_uri} rdf:type {class_uri}")
                g.add((instance_uri, RDF.type, class_uri))
                
                # Add name and description
                g.add((instance_uri, RDFS.label, Literal(name)))
                if description:
                    g.add((instance_uri, SKOS.definition, Literal(description)))
            
            # Process all edges
            edge_count = await graph_storage.edge_count()
            print(f"\nTotal edges in graph: {edge_count}")
            
            # Get the relationships-to-chunks map to access source text
            rel_to_chunk_map = grag.state_manager._relationships_to_chunks
            
            for i in range(edge_count):
                edge = await graph_storage.get_edge_by_index(i)
                if edge is None:
                    continue
                    
                source = edge.source
                target = edge.target
                description = edge.description
                property_val = edge.property
                chunks = edge.chunks  # List of chunk hashes
                
                print(f"Edge: {source} -> {target}")
                print(f"Description: {description}")
                print(f"Property: {property_val}")
                print(f"Chunks: {chunks}")
                
                # Check if we have URIs for both source and target
                if source in node_uris and target in node_uris:
                    source_uri = node_uris[source]
                    target_uri = node_uris[target]
                    
                    # Default predicate
                    predicate = SE.relatedTo
                    
                    # Try to resolve the property CURIE
                    if property_val:
                        logger.info(f"\nProcessing edge property: {property_val}")
                        resolved = resolver(property_val)
                        if resolved is not None:
                            predicate = resolved
                            # Check if this is a new property
                            if not any(ontology.triples((resolved, RDF.type, None))):
                                g.add((resolved, RDF.type, SE.NewProperty))
                    
                    print(f"Adding triple: {source_uri} {predicate} {target_uri}")
                    g.add((source_uri, predicate, target_uri))
                    
                    if description:
                        print(f"Adding relationship description: {description}")
                        g.add((source_uri, SKOS.note, Literal(description)))
                    
                    # Process chunks for this relationship
                    if chunks:
                        try:
                            # Create a better structured evidence URI
                            # Take just the local part of the target (without the full namespace)
                            source_local = source_uri.split('/')[-1]
                            target_local = target_uri.split('/')[-1]
                            
                            # Create a properly structured evidence node
                            # Format: source entity + "evidence" + target local name
                            evidence_node = URIRef(f"{source_uri}/evidence/{target_local}")
                            g.add((evidence_node, RDF.type, SE.Evidence))
                            g.add((source_uri, SE.hasEvidence, evidence_node))
                            
                            # Process each chunk ID
                            for j, chunk_id in enumerate(chunks):
                                try:
                                    # Create a cleaner chunk URI as a child of the evidence node
                                    # Format: evidence URI + "chunk" + index number
                                    chunk_uri = URIRef(f"{evidence_node}/chunk/{j}")
                                    g.add((evidence_node, SE.hasChunk, chunk_uri))
                                    g.add((chunk_uri, RDF.type, SE.TextChunk))
                                    g.add((chunk_uri, SE.chunkId, Literal(str(chunk_id), datatype=XSD.string)))
                                    
                                    # Get a single chunk from storage - using the right API call
                                    # The get method expects an iterable of keys and returns an iterable of values
                                    chunk_objects = await chunk_storage.get([chunk_id])
                                    chunk_obj = next(iter(chunk_objects), None)
                                    
                                    if chunk_obj is not None:
                                        # Add the actual content from the chunk
                                        chunk_content = chunk_obj.content
                                        g.add((chunk_uri, SE.text, Literal(chunk_content)))
                                        print(f"Added chunk content: {chunk_content[:50]}...")
                                    else:
                                        print(f"Chunk {chunk_id} not found in storage")
                                        g.add((chunk_uri, RDFS.comment, Literal(f"Chunk ID {chunk_id} not found in storage")))
                                except Exception as e:
                                    logger.warning(f"Error processing chunk {chunk_id}: {e}")
                                    g.add((chunk_uri, RDFS.comment, Literal(f"Error: {str(e)}")))
                                    # Continue with next chunk even if this one fails
                        except Exception as e:
                            logger.error(f"Error processing chunks for relationship {source}->{target}: {e}")
            
            return g
        except Exception as e:
            logger.error(f"Error building RDF graph: {e}")
            # Return an empty graph if there was an error
            return Graph()
        
    # Run the async function to build the graph
    try:
        g = get_event_loop().run_until_complete(build_rdf_graph())
    except Exception as e:
        logger.error(f"Error running async graph builder: {e}")
        g = Graph()

    # Apply hygiene rules and save RDF graph
    g = apply_hygiene_rules(g, ontology)
    
    # Save a local copy for inspection
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    local_output_path = os.path.join(output_dir, "Printer3D_knowledge.ttl")
    g.serialize(destination=local_output_path, format='turtle')
    print(f"\nSaved local RDF output to {local_output_path}")
    
    # Serialize the graph to a BytesIO object in turtle format for upload
    data_in_memory = BytesIO()
    g.serialize(destination=data_in_memory, format='turtle')
    data_in_memory.seek(0)  # Necessary to rewind the BytesIO object

    # Upload to data.world if a version is specified
    if args.version:
        # Your DW_AUTH_TOKEN should be available throughout the execution
        api_token = os.getenv('DW_AUTH_TOKEN')
        if not api_token:
            logger.error("DW_AUTH_TOKEN not found in environment variables. Skipping data.world upload.")
        else:
            filename = "ontology.ttl"
            url = f"https://api.data.world/v0/uploads/{args.version}/files/{filename}"
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
            
            print(f"Uploading to data.world URL: {url}")
            try:
                response = requests.put(
                    url,
                    headers=headers,
                    data=data_in_memory  # Use the BytesIO object as your data
                )
                
                if response.status_code == 200: 
                    print("\nSuccessfully uploaded RDF to data.world")
                else: 
                    print(f"\nFailed to upload to data.world: {response.text}")
            except Exception as e:
                print(f"\nError during upload to data.world: {e}")
    else:
        print("\nNo version specified, skipping data.world upload")
        


if __name__ == "__main__":
    main()
