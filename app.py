import os
import logging
import base64
import shutil
import requests
import traceback
import datetime
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory, send_file
from fast_graphrag import GraphRAG
from fast_graphrag._utils import get_event_loop
from rdflib import Graph, URIRef, Literal, Dataset, Namespace
from rdflib.namespace import RDF, OWL, RDFS, SKOS, XSD
from urllib.parse import urlparse
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for ontology data
prefixes: Dict[str, str] = {}
domain: str = ""
entity_types: list[str] = []
ontology: Graph = Graph()
curie_map: Dict[str, URIRef] = {}  # Case-insensitive CURIE lookup

# Track last successful upload for each version
last_successful_uploads = {}

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

    # ontology_data is the direct input now, so no need to read files
    domain = ontology_data 
    temp: Graph = Graph()
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
    class_count = 0
    
    for class_uri in ontology.subjects(RDF.type, OWL.Class):
        class_count += 1
        label_count = 0
        
        # Get labels for this class
        for label in ontology.objects(class_uri, RDFS.label):
            label_count += 1
            logger.info(f"Found Class: {class_uri}, Label: {label}")
            entity_types.append(str(label))
        
        # Log if no labels found
        if label_count == 0:
            logger.warning(f"⚠️ Class has no rdfs:label: {class_uri}")
    
    # Log summary
    logger.info(f"Found {class_count} OWL Classes and extracted {len(entity_types)} labels")
    logger.info(f"Entity types: {entity_types}")

    # Build case-insensitive CURIE map
    curie_count = 0
    for s, p, o in ontology:
        if isinstance(s, URIRef):
            # Add mapping from full IRI (new)
            full_iri = str(s).lower()
            curie_map[full_iri] = s
            logger.info(f"Adding mapping: full IRI '{full_iri}' → <{s}>")
            curie_count += 1
            
            # Add mappings from labels
            for label in ontology.objects(s, RDFS.label):
                curie_map[label.lower()] = s
                logger.info(f"Adding mapping: label '{label.lower()}' → <{s}>")
                curie_count += 1
            
            # Add mapping from local part
            local_part = get_local_part(s).lower()
            curie_map[local_part] = s
            logger.info(f"Adding mapping: local part '{local_part}' → <{s}>")
            curie_count += 1
            
            # Add mapping from CURIEs
            for prefix, uri in prefixes.items():
                if str(s).startswith(str(uri)):
                    local = str(s)[len(str(uri)):]
                    curie = f"{prefix}:{local}"
                    curie_map[curie.lower()] = s
                    logger.info(f"Adding mapping: CURIE '{curie.lower()}' → <{s}>")
                    curie_count += 1
    
    logger.info(f"Added {curie_count} mappings to curie_map")
    logger.info(f"First 10 keys in curie_map: {list(curie_map.keys())[:10]}")
                    
def make_curie_resolver(prefixes: Dict[str, str], curie_map: Dict[str, URIRef]):
    """Create a closure that captures the prefix and curie mappings"""
    def resolve_curie(curie: str) -> Optional[URIRef]:
        """Resolve a CURIE using case-insensitive lookup."""
        logger.info(f"Resolving CURIE: {curie}")
        
        if not curie:
            logger.warning("Empty CURIE provided")
            return None
            
        if ':' not in curie:
            logger.warning(f"Invalid CURIE format (no colon): {curie}")
            return None
        
        # Try case-insensitive lookup first
        curie_lower = curie.lower()
        if curie_lower in curie_map:
            uri = curie_map[curie_lower]
            logger.info(f"✅ Found exact match in curie_map: '{curie_lower}' → {uri}")
            return uri
        
        # Try additional case-insensitive matching strategies
        prefix, local = curie.split(':', 1)
        prefix_lower = prefix.lower()
        local_lower = local.lower()
        
        # Check if just the local part exists (without prefix)
        if local_lower in curie_map:
            uri = curie_map[local_lower]
            logger.info(f"✅ Matched local part only: '{local_lower}' → {uri}")
            return uri
            
        # Try to resolve using available prefixes
        available_prefixes = [p.lower() for p in prefixes]
        logger.info(f"Checking prefix '{prefix_lower}' against available prefixes: {available_prefixes}")
        
        if prefix_lower in available_prefixes:
            # Find the original case of the prefix
            original_prefix = next(p for p in prefixes if p.lower() == prefix_lower)
            uri_prefix = prefixes[original_prefix]
            
            # First try to find the full expanded URI in the curie_map
            full_uri_lower = (uri_prefix + safe_uri_name(local)).lower()
            if full_uri_lower in curie_map:
                uri = curie_map[full_uri_lower]
                logger.info(f"✅ Matched expanded URI: '{full_uri_lower}' → {uri}")
                return uri
                
            # If not found, construct a new URI
            uri = URIRef(uri_prefix + safe_uri_name(local))
            logger.info(f"✅ Constructed new URI from prefix: {prefix} → {uri}")
            return uri
        
        # Last resort: try to find any similar keys
        logger.warning(f"❌ Not found in curie_map: '{curie_lower}'")
        similar_keys = [k for k in curie_map.keys() if 
                        (curie_lower in k or k in curie_lower or
                         local_lower in k or k in local_lower)]
        if similar_keys:
            logger.info(f"Similar keys that might match: {similar_keys[:5]}")
            
            # If we have very similar keys, use the first one as a last resort
            if len(similar_keys) > 0 and (local_lower in similar_keys[0] or similar_keys[0] in local_lower):
                uri = curie_map[similar_keys[0]]
                logger.info(f"⚠️ Using closest match as fallback: '{similar_keys[0]}' → {uri}")
                return uri
        
        logger.warning(f"❌ Failed to resolve CURIE: {curie}")
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

async def build_rdf_graph(grag):
    try:
        # Process nodes from the graph storage
        graph_storage = grag.state_manager.graph_storage
        chunk_storage = grag.state_manager.chunk_storage
        
        # Create RDF graph
        g = Graph()
        g.bind("gr", SE)  # 'gr' for GraphRAG
        
        # Copy prefixes from ontology
        for prefix, uri in prefixes.items():
            g.bind(prefix, uri)
            
        node_count = await graph_storage.node_count()
        logger.info(f"Total nodes in graph: {node_count}")
        
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
            
            logger.debug(f"Node: {name}")
            logger.debug(f"Type: {node_type}")
            logger.debug(f"Description: {description}")
            
            # Create URI for this entity
            instance_uri = SE[safe_uri_name(name)]
            node_uris[name] = instance_uri
            node_names[i] = name
            
            # Create resolver with context
            resolver = make_curie_resolver(prefixes, curie_map)
            
            # Try to resolve type as CURIE first
            logger.info(f"Trying to resolve type for node: {name}, type: {node_type}")
            logger.info(f"Available entity_types: {entity_types}")
            
            if node_type and ':' in node_type:
                logger.info(f"Attempting to resolve as CURIE: {node_type}")
                resolved_type = resolver(node_type)
                if resolved_type is not None:
                    class_uri = resolved_type
                    logger.info(f"✅ Resolved CURIE to: {class_uri}")
                else:
                    logger.warning(f"❌ Failed to resolve CURIE: {node_type}")
                    class_uri = SE.UNKNOWN
                    
            # Then try matching against entity type labels
            elif node_type and node_type.lower() in [et.lower() for et in entity_types]:
                logger.info(f"Node type '{node_type}' found in entity_types")
                
                # Log all lower-cased entity types for comparison
                logger.info(f"Entity types (lowercase): {[et.lower() for et in entity_types]}")
                
                # Debug the curie_map contents
                logger.info(f"Looking up '{node_type.lower()}' in curie_map")
                logger.info(f"curie_map keys: {list(curie_map.keys())[:10]}... (showing first 10)")
                
                class_uri = curie_map.get(node_type.lower(), None)
                if class_uri is None:
                    logger.warning(f"❌ Node type {node_type} resulted in None for CURIE resolution")
                    # Try to find similar keys
                    similar_keys = [k for k in curie_map.keys() if node_type.lower() in k or k in node_type.lower()]
                    if similar_keys:
                        logger.info(f"Similar keys in curie_map: {similar_keys}")
                    class_uri = SE.UNKNOWN
                else:
                    logger.info(f"✅ Successfully mapped '{node_type}' to {class_uri}")
            else:
                logger.warning(f"❌ No match for node type: {node_type}")
                class_uri = SE.UNKNOWN
                
            if class_uri == SE.UNKNOWN:
                g.add((class_uri, RDF.type, OWL.Class))  # Define UNKNOWN as a class if not already done
                
            logger.debug(f"Adding triple: {instance_uri} rdf:type {class_uri}")
            g.add((instance_uri, RDF.type, class_uri))
            
            # Add name and description
            g.add((instance_uri, RDFS.label, Literal(name)))
            if description:
                g.add((instance_uri, SKOS.definition, Literal(description)))
        
        # Process all edges
        edge_count = await graph_storage.edge_count()
        logger.info(f"Total edges in graph: {edge_count}")
        
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
            
            logger.debug(f"Edge: {source} -> {target}")
            logger.debug(f"Description: {description}")
            logger.debug(f"Property: {property_val}")
            logger.debug(f"Chunks: {chunks}")
            
            # Check if we have URIs for both source and target
            if source in node_uris and target in node_uris:
                source_uri = node_uris[source]
                target_uri = node_uris[target]
                
                # Default predicate
                predicate = SE.relatedTo
                
                # Try to resolve the property CURIE
                if property_val:
                    logger.debug(f"Processing edge property: {property_val}")
                    resolved = resolver(property_val)
                    if resolved is not None:
                        predicate = resolved
                        # Check if this is a new property
                        if not any(ontology.triples((resolved, RDF.type, None))):
                            g.add((resolved, RDF.type, SE.NewProperty))
                
                logger.debug(f"Adding triple: {source_uri} {predicate} {target_uri}")
                g.add((source_uri, predicate, target_uri))
                
                if description:
                    logger.debug(f"Adding relationship description: {description}")
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
                                    logger.debug(f"Added chunk content: {chunk_content[:50]}...")
                                else:
                                    logger.warning(f"Chunk {chunk_id} not found in storage")
                                    g.add((chunk_uri, RDFS.comment, Literal(f"Chunk ID {chunk_id} not found in storage")))
                            except Exception as e:
                                logger.warning(f"Error processing chunk {chunk_id}: {e}")
                                g.add((chunk_uri, RDFS.comment, Literal(f"Error: {str(e)}")))
                                # Continue with next chunk even if this one fails
                    except Exception as e:
                        logger.error(f"Error processing chunks for relationship {source}->{target}: {e}")
        
        return g
    except Exception as e:
        logger.error(f"Error building RDF graph: {e}", exc_info=True)
        # Return an empty graph if there was an error
        return Graph()

# Routes
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/debug', methods=['GET'])
def debug_page():
    """A simple page to test if JavaScript is working"""
    return """
    <html>
    <head>
        <title>Debug Page</title>
        <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            button { padding: 10px; margin: 10px 0; }
            .indicator { display: none; color: red; font-weight: bold; }
        </style>
        <script>
            function testButton() {
                document.getElementById('testBtn').disabled = true;
                document.getElementById('indicator').style.display = 'block';
                return false;
            }
        </script>
    </head>
    <body>
        <h1>Debug Page</h1>
        <p>This page tests if JavaScript is working properly.</p>
        
        <form onsubmit="return testButton()">
            <button id="testBtn" type="submit">Test Button</button>
        </form>
        
        <div id="indicator" class="indicator">
            JavaScript is working! This text should appear when you click the button.
        </div>
    </body>
    </html>
    """

@app.route('/simple_upload', methods=['GET'])
def simple_upload():
    """A simplified version of the upload page for testing timeout issues"""
    version = request.args.get('version', '')
    return f"""
    <html>
    <head>
        <title>Simple Upload Page for {version}</title>
    </head>
    <body>
        <h1>Simple Upload Page for {version}</h1>
        <p>This is a simplified page for testing.</p>
    </body>
    </html>
    """

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    # For debugging
    logger.info(f"Received {'POST' if request.method == 'POST' else 'GET'} request to /upload")
    
    version = request.args.get('version', '')
    logger.info(f"Version from args: {version}")
    
    if request.method == 'POST':
        # Handle form submission
        input_text = request.form.get('input_text', '')
        version = request.form.get('version', '')
        
        if not input_text:
            return """
            <html>
            <head>
                <title>Upload page for version {}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: red; font-weight: bold; }}
                    textarea {{ width: 100%; height: 75vh; margin-top: 10px; padding: 10px; }}
                    button {{ padding: 10px 20px; background-color: #4CAF50; color: white; 
                             border: none; cursor: pointer; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>Upload page for version {}</h1>
                <div class="error">Error: Please provide input text</div>
                <form method="post">
                    <input type="hidden" name="version" value="{}">
                    <button type="submit">Submit</button>
                    <label for="input_text">Input text:</label>
                    <textarea name="input_text" id="input_text"></textarea>
                </form>
            </body>
            </html>
            """.format(version, version, version)
            
        # Process the data using the API endpoint logic
        try:
            # Create a resolver with context
            resolver = make_curie_resolver(prefixes, curie_map)
            
            # If renew flag is set, clean up old files (always true for this endpoint)
            renew = True
            if renew:
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
                            return f"""
                            <html>
                            <head>
                                <title>Error</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                    h1 {{ color: #333; }}
                                    .error {{ color: red; }}
                                </style>
                            </head>
                            <body>
                                <h1>Error</h1>
                                <div class="error">Failed to clean up files: {str(e)}</div>
                                <a href="/upload?version={version}">Go Back</a>
                            </body>
                            </html>
                            """
                    logger.info("Cleanup complete")
            
            # Load ontology
            try:
                # Download data from the data.world API
                if version:
                    api_token = os.getenv('DW_AUTH_TOKEN')
                    if not api_token:
                        return f"""
                        <html>
                        <head>
                            <title>Error</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                h1 {{ color: #333; }}
                                .error {{ color: red; }}
                            </style>
                        </head>
                        <body>
                            <h1>Error</h1>
                            <div class="error">DW_AUTH_TOKEN not found in environment variables</div>
                            <a href="/upload?version={version}">Go Back</a>
                        </body>
                        </html>
                        """
                        
                    headers = {"Authorization": f"Bearer {api_token}"}
                    url = f"https://api.data.world/v0/file_download/{version}/ontology.ttl"
                    logger.info(f"Downloading ontology from {url}")
                    response = requests.get(url, headers=headers)
                    if not response.ok:
                        response.raise_for_status()
                    ontology_data = response.text
                    load_ontology(ontology_data)
                else:
                    # If no version specified, read from local ontology.ttl file
                    try:
                        with open('ontology.ttl', 'r') as file:
                            ontology_data = file.read()
                            load_ontology(ontology_data)
                    except FileNotFoundError:
                        return f"""
                        <html>
                        <head>
                            <title>Error</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                h1 {{ color: #333; }}
                                .error {{ color: red; }}
                            </style>
                        </head>
                        <body>
                            <h1>Error</h1>
                            <div class="error">Local ontology.ttl file not found</div>
                            <a href="/upload?version={version}">Go Back</a>
                        </body>
                        </html>
                        """
                        
                logger.info("Loaded ontology")
                logger.info(f"Found prefixes: {prefixes}")
                logger.info(f"Found entity types: {entity_types}")
            except (FileNotFoundError, ValueError) as e:
                return f"""
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .error {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Error</h1>
                    <div class="error">Failed to load ontology: {str(e)}</div>
                    <a href="/upload?version={version}">Go Back</a>
                </body>
                </html>
                """
            except requests.HTTPError as e:
                return f"""
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .error {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Error</h1>
                    <div class="error">Failed to download ontology: {str(e)}</div>
                    <a href="/upload?version={version}">Go Back</a>
                </body>
                </html>
                """
            
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
                    resolve_curie=resolver
                )
            )
            
            logger.info(f"Using provided input text ({len(input_text)} characters)")
            logger.info("Starting content insertion...")
            grag.insert(input_text)
            logger.info("Completed content insertion")
            
            # Build and save RDF graph
            g = get_event_loop().run_until_complete(build_rdf_graph(grag))
            
            # Apply hygiene rules and save RDF graph
            g = apply_hygiene_rules(g, ontology)
            
            # Extract version name for the filename (part after the slash)
            if version and '/' in version:
                version_part = version.split('/')[-1]
                filename = f"{version_part}.ontology.ttl"
            else:
                filename = "unknown-version.ontology.ttl"
                
            # Save a local copy for inspection
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            local_output_path = os.path.join(output_dir, filename)
            g.serialize(destination=local_output_path, format='turtle')
            logger.info(f"Saved local RDF output to {local_output_path}")
            
            # Serialize the graph to a BytesIO object in turtle format for upload
            data_in_memory = BytesIO()
            g.serialize(destination=data_in_memory, format='turtle')
            data_in_memory.seek(0)  # Necessary to rewind the BytesIO object

            # Upload to data.world if a version is specified
            upload_status = "Not uploaded (no version specified)"
            if version:
                # Your DW_AUTH_TOKEN should be available throughout the execution
                api_token = os.getenv('DW_AUTH_TOKEN')
                if not api_token:
                    return f"""
                    <html>
                    <head>
                        <title>Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1 {{ color: #333; }}
                            .error {{ color: red; }}
                        </style>
                    </head>
                    <body>
                        <h1>Error</h1>
                        <div class="error">DW_AUTH_TOKEN not found for upload</div>
                        <a href="/upload?version={version}">Go Back</a>
                    </body>
                    </html>
                    """
                    
                filename = "ontology.ttl"
                url = f"https://api.data.world/v0/uploads/{version}/files/{filename}"
                
                headers = {
                    "accept": "application/json",
                    "Authorization": f"Bearer {api_token}"
                }
                
                logger.info(f"Uploading to data.world URL: {url}")
                try:
                    response = requests.put(
                        url,
                        headers=headers,
                        data=data_in_memory
                    )
                    
                    if response.status_code == 200:
                        logger.info("Successfully uploaded RDF to data.world")
                        upload_status = "Successfully uploaded to data.world"
                        # Record successful upload time
                        last_successful_uploads[version] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        error_message = response.text
                        try:
                            # Attempt to parse error as JSON
                            error_json = response.json()
                            if 'message' in error_json:
                                error_message = error_json['message']
                        except:
                            # If not JSON, use text as is
                            pass
                            
                        logger.error(f"Failed to upload to data.world: {error_message}")
                        upload_status = f"Failed to upload (HTTP {response.status_code}): {error_message}"
                except Exception as e:
                    logger.error(f"Error during upload to data.world: {e}")
                    upload_status = f"Error during upload: {str(e)}"
            else:
                logger.info("No version specified, skipping data.world upload")
            
            # Return result page - either success or failure
            is_success = "Failed to upload" not in upload_status and "Error during upload" not in upload_status
            status_class = "success" if is_success else "error"
            status_color = "green" if is_success else "red"
            status_icon = "✅" if is_success else "❌"
            
            # Get last successful submission time if available
            last_success_info = ""
            if version in last_successful_uploads and not is_success:
                last_time = last_successful_uploads[version]
                last_success_info = f'''
                <div class="info-banner">
                    <p><span class="success-icon">ℹ️</span> Last successful submission: {last_time}</p>
                </div>
                '''
            
            # Get just the filename part for the download link
            filename = os.path.basename(local_output_path)
            
            return f"""
            <html>
            <head>
                <title>Processing Complete</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .success {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .error {{ color: red; font-weight: bold; }}
                    pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    .status {{ color: {status_color}; font-weight: bold; }}
                    .status-banner {{
                        background-color: {("#dff2bf" if status_color == "green" else "#ffbaba")};
                        color: {status_color};
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    .info-banner {{
                        background-color: #e7f3fe;
                        color: #0c5460;
                        padding: 10px;
                        margin-bottom: 15px;
                        border-radius: 5px;
                    }}
                    .success-icon {{ font-size: 1.2em; margin-right: 5px; }}
                    .button {{
                        display: inline-block;
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 15px;
                        margin: 10px 5px 10px 0;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        text-decoration: none;
                    }}
                    .button:hover {{
                        background-color: #45a049;
                    }}
                </style>
            </head>
            <body>
                <h1>Processing Complete</h1>
                
                <div class="status-banner">
                    <span class="success-icon">{status_icon}</span> {upload_status}
                </div>
                
                {last_success_info}
                
                <p>Your text has been processed and saved as <strong>{filename}</strong>.</p>
                
                <div>
                    <a href="/download/{filename}" class="button">Download Result</a>
                    <a href="/upload?version={version}" class="button">Process Another Document</a>
                </div>
            </body>
            </html>
            """
            
        except Exception as e:
            return f"""
            <html>
            <head>
                <title>Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: red; }}
                    pre {{ background-color: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <h1>Error</h1>
                <div class="error">Failed to process content: {str(e)}</div>
                <pre>{traceback.format_exc()}</pre>
                <a href="/upload?version={version}">Go Back</a>
            </body>
            </html>
            """
    
    # Display the upload form (GET request)
    logger.info("Preparing to render upload form")
    last_upload_info = ""
    if version in last_successful_uploads:
        last_time = last_successful_uploads[version]
        logger.info(f"Found last successful upload for version {version}: {last_time}")
        last_upload_info = f'''
        <div class="success-banner">
            <span class="success-icon">✅</span> Last successful submission: {last_time}
        </div>
        '''
    else:
        logger.info(f"No previous successful uploads found for version {version}")
    
    return f"""
    <html>
    <head>
        <title>Upload page for version {version}</title>
        <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            textarea {{ width: 100%; height: 70vh; margin-top: 10px; padding: 10px; }}
            button {{ padding: 10px 20px; background-color: #4CAF50; color: white; 
                     border: none; cursor: pointer; margin-bottom: 10px; }}
            .success-banner {{ 
                background-color: #dff2bf; 
                color: #4F8A10; 
                padding: 10px; 
                margin-bottom: 15px; 
                border-radius: 5px; 
                font-weight: bold;
            }}
            .success-icon {{ font-size: 1.2em; margin-right: 5px; }}
            .error-banner {{ 
                background-color: #ffbaba; 
                color: #d8000c; 
                padding: 10px; 
                margin-bottom: 15px; 
                border-radius: 5px; 
                font-weight: bold;
            }}
            .processing {{ 
                display: none;
                color: #0066cc; 
                font-weight: bold;
                margin-left: 15px;
                vertical-align: middle;
            }}
            .spinner {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0,102,204,.3);
                border-radius: 50%;
                border-top-color: #0066cc;
                animation: spin 1s ease-in-out infinite;
                vertical-align: middle;
                margin-right: 10px;
            }}
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        <script>
            function showProcessing() {{
                // Simple change - just make the button say "Processing..."
                document.getElementById('submitBtn').value = "Processing...";
                
                // Show the processing indicator
                document.getElementById('processingIndicator').style.display = "inline";
                
                return true;
            }}
        </script>
    </head>
    <body>
        <h1>Upload page for version {version}</h1>
        {last_upload_info}
        <form method="post" onsubmit="return showProcessing()">
            <input type="hidden" name="version" value="{version}">
            <input id="submitBtn" type="submit" value="Submit" style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">
            <span id="processingIndicator" style="display: none; margin-left: 10px; color: #666;">Processing...</span>
            <label for="input_text">Input text:</label>
            <textarea name="input_text" id="input_text"></textarea>
        </form>
    </body>
    </html>
    """

@app.route('/process', methods=['POST'])
def process():
    if not request.json:
        return jsonify({"error": "Missing JSON payload"}), 400
    
    version = request.json.get('version')
    renew = request.json.get('renew', False)
    input_text = request.json.get('input_text', None)
    
    logger.info(f"Received request with version={version}, renew={renew}, input_text_provided={input_text is not None}")
    
    # If renew flag is set, clean up old files
    if renew:
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
                    return jsonify({"error": f"Failed to clean up files: {str(e)}"}), 500
            logger.info("Cleanup complete")

    # Load ontology
    try:
        # Download data from the data.world API
        if version:
            api_token = os.getenv('DW_AUTH_TOKEN')
            if not api_token:
                return jsonify({"error": "DW_AUTH_TOKEN not found in environment variables"}), 500
                
            headers = {"Authorization": f"Bearer {api_token}"}
            url = f"https://api.data.world/v0/file_download/{version}/ontology.ttl"
            logger.info(f"Downloading ontology from {url}")
            response = requests.get(url, headers=headers)
            if not response.ok:
                response.raise_for_status()
            ontology_data = response.text
            load_ontology(ontology_data)
        else:
            # If no version specified, read from local ontology.ttl file
            try:
                with open('ontology.ttl', 'r') as file:
                    ontology_data = file.read()
                    load_ontology(ontology_data)
            except FileNotFoundError:
                return jsonify({"error": "Local ontology.ttl file not found"}), 404
                
        logger.info("Loaded ontology")
        logger.info(f"Found prefixes: {prefixes}")
        logger.info(f"Found entity types: {entity_types}")
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": f"Failed to load ontology: {str(e)}"}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"Failed to download ontology: {str(e)}"}), 500

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
    try:
        if not input_text:
            return jsonify({"error": "Missing input_text field in request"}), 400
            
        # Use the input text from the request
        logger.info(f"Using provided input text ({len(input_text)} characters)")
        content = input_text
        
        logger.info("Starting content insertion...")
        grag.insert(content)
        logger.info("Completed content insertion")
    except Exception as e:
        return jsonify({"error": f"Failed to process content: {str(e)}"}), 500

    # Build and save RDF graph
    try:
        # Run the async function to build the graph
        g = get_event_loop().run_until_complete(build_rdf_graph(grag))
        
        # Apply hygiene rules and save RDF graph
        g = apply_hygiene_rules(g, ontology)
        
        # Extract version name for the filename (part after the slash)
        if version and '/' in version:
            version_part = version.split('/')[-1]
            filename = f"{version_part}.ontology.ttl"
        else:
            filename = "unknown-version.ontology.ttl"
            
        # Save a local copy for inspection
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, filename)
        g.serialize(destination=local_output_path, format='turtle')
        logger.info(f"Saved local RDF output to {local_output_path}")
        
        # Serialize the graph to a BytesIO object in turtle format for upload
        data_in_memory = BytesIO()
        g.serialize(destination=data_in_memory, format='turtle')
        data_in_memory.seek(0)  # Necessary to rewind the BytesIO object

        # Upload to data.world if a version is specified
        if version:
            # Your DW_AUTH_TOKEN should be available throughout the execution
            api_token = os.getenv('DW_AUTH_TOKEN')
            if not api_token:
                return jsonify({"error": "DW_AUTH_TOKEN not found for upload"}), 500
                
            filename = "ontology.ttl"
            url = f"https://api.data.world/v0/uploads/{version}/files/{filename}"
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
            
            logger.info(f"Uploading to data.world URL: {url}")
            try:
                response = requests.put(
                    url,
                    headers=headers,
                    data=data_in_memory  # Use the BytesIO object as your data
                )
                
                if response.status_code == 200: 
                    logger.info("Successfully uploaded RDF to data.world")
                else: 
                    logger.error(f"Failed to upload to data.world: {response.text}")
                    return jsonify({"error": f"Failed to upload to data.world: {response.text}"}), 500
            except Exception as e:
                logger.error(f"Error during upload to data.world: {e}")
                return jsonify({"error": f"Error during upload: {str(e)}"}), 500
        else:
            logger.info("No version specified, skipping data.world upload")
            
        return jsonify({
            "status": "success", 
            "message": "Processing completed successfully",
            "local_output": local_output_path,
            "uploaded_to_dataworld": bool(version)
        })
        
    except Exception as e:
        logger.error(f"Error in graph processing: {e}", exc_info=True)
        return jsonify({"error": f"Failed to build RDF graph: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "Missing question field"}), 400
    
    # Create GraphRAG instance
    try:
        # Create resolver with context
        resolver = make_curie_resolver(prefixes, curie_map)
        
        grag = GraphRAG(
            working_dir="./print3D_example",
            domain=domain,
            entity_types=entity_types,
            config=GraphRAG.Config(
                resolve_curie=resolver
            )
        )
        
        question = request.json['question']
        with_references = request.json.get('with_references', False)
        
        result = grag.query(question, with_references=with_references)
        return jsonify({
            "response": result.response,
            "references": result.references if with_references else None
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a processed ontology file"""
    # Security check - only allow ttl files from the output directory
    if not filename.endswith('.ttl'):
        return "Invalid file type", 400
        
    # Check for path traversal attempts
    if '..' in filename or '/' in filename:
        return "Invalid filename", 400
        
    # Send the file from the output directory
    output_dir = os.path.abspath("./output")
    return send_from_directory(output_dir, filename, as_attachment=True)

@app.route('/workshop', methods=['GET'])
def workshop_page():
    """Display a list of projects for a given organization with optional filtering"""
    org = request.args.get('org', '')
    filter_text = request.args.get('filter', '')
    
    if not org:
        return "Please specify an organization with the 'org' parameter", 400
    
    # Get the API token from environment
    api_token = os.getenv('DW_AUTH_TOKEN')
    if not api_token:
        return "API token not found in environment variables", 500
    
    # Fetch projects from the data.world API
    url = f"https://api.data.world/v0/projects/{org}?limit=100"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if not response.ok:
            response.raise_for_status()
        
        projects_data = response.json()
        
        # Build the HTML response
        projects_html = ""
        project_count = 0
        total_projects = len(projects_data.get('records', []))
        
        for project in projects_data.get('records', []):
            project_id = project.get('id')
            project_title = project.get('title', project_id)
            
            # Apply filter if specified
            if filter_text and filter_text.lower() not in project_title.lower():
                continue
                
            project_count += 1
            upload_url = f"https://kgc-ddw-entity-db8350081a81.herokuapp.com/upload?version={org}/{project_id}"
            projects_html += f'<li><a href="{upload_url}" target="_blank">{project_title}</a></li>\n'
        
        # Filter status message
        filter_message = ""
        if filter_text:
            filter_message = f"""
            <div class="filter-status">
                <p>Showing {project_count} out of {total_projects} projects containing "{filter_text}"</p>
                <p><a href="/workshop?org={org}" class="reset-filter">Show all projects</a></p>
            </div>
            """
        
        # Filter form
        filter_form = f"""
        <div class="filter-form">
            <form method="get" action="/workshop">
                <input type="hidden" name="org" value="{org}">
                <label for="filter">Filter projects:</label>
                <input type="text" id="filter" name="filter" value="{filter_text}" placeholder="Enter filter text">
                <button type="submit">Apply Filter</button>
            </form>
        </div>
        """
        
        return f"""
        <html>
        <head>
            <title>Workshop Projects for {org}</title>
            <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" type="image/png">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #333; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin-bottom: 10px; padding: 8px; border-radius: 4px; }}
                li:hover {{ background-color: #f5f5f5; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .filter-form {{ margin: 20px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                .filter-form input[type="text"] {{ padding: 8px; width: 300px; margin-right: 10px; }}
                .filter-form button {{ padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                .filter-form button:hover {{ background-color: #45a049; }}
                .filter-status {{ margin-bottom: 20px; color: #666; }}
                .reset-filter {{ color: #666; text-decoration: underline; }}
                .no-results {{ padding: 20px; background-color: #f8f8f8; border-radius: 5px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Workshop Projects for {org}</h1>
                <p>Click on a project to open the upload page in a new tab.</p>
            </div>
            
            {filter_form}
            {filter_message}
            
            {('<ul>' + projects_html + '</ul>') if project_count > 0 else '<div class="no-results">No projects match your filter criteria.</div>'}
        </body>
        </html>
        """
    except requests.HTTPError as e:
        return f"Error fetching projects: {str(e)}", 500
    except Exception as e:
        logger.error(f"Error in workshop page: {e}", exc_info=True)
        return f"Error processing workshop data: {str(e)}", 500

# We're now using data URLs directly in HTML for the favicon

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)