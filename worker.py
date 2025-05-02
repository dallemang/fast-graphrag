#!/usr/bin/env python3
"""
Worker process that handles background processing for GraphRAG.
This worker connects to Redis and processes jobs from the queue.
"""

import os
import sys
import logging
import base64
import shutil
import requests
import traceback
import datetime
import ssl
from io import BytesIO
from worker_utils import ProgressReporter
from retry_utils import retry_on_exception
from rdflib import Graph, URIRef, Literal, Dataset, Namespace
from rdflib.namespace import RDF, OWL, RDFS, SKOS, XSD
from urllib.parse import urlparse
from typing import Dict, Optional
from fast_graphrag import GraphRAG
from fast_graphrag._utils import get_event_loop

# Import Redis and RQ related libraries
from redis import Redis
# Just import the necessary RQ objects
from rq import Worker, Queue

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Global variables for ontology data
prefixes: Dict[str, str] = {}
domain: str = ""
entity_types: list[str] = []
ontology: Graph = Graph()
curie_map: Dict[str, URIRef] = {}  # Case-insensitive CURIE lookup

# System namespace for GraphRAG
SE = Namespace("https://models.data.world/graphrag/")  # type: ignore

# Define the Redis connection
def get_redis_connection():
    # First, try to use UPSTASH_REDIS_URL
    upstash_url = os.environ.get('UPSTASH_REDIS_URL')
    if upstash_url:
        logger.info(f"Using Upstash Redis URL: {upstash_url}")
        try:
            connection = Redis.from_url(
                upstash_url,
                socket_timeout=60,
                socket_connect_timeout=30,
                ssl_cert_reqs=None,
                ssl_check_hostname=False,
                decode_responses=False  # Important for binary data
            )
            connection.ping()  # Test the connection
            logger.info("Successfully connected to Upstash Redis using URL")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect using Upstash Redis URL: {e}")
    
    # Fall back to direct connection parameters
    host = os.environ.get('REDIS_HOST', 'verified-stag-21591.upstash.io')
    port = int(os.environ.get('REDIS_PORT', '6379'))
    password = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '')
    
    logger.info(f"Falling back to direct Upstash connection at {host}:{port}")
    try:
        # Create connection with SSL settings
        connection = Redis(
            host=host,
            port=port,
            password=password,
            ssl=True,
            ssl_cert_reqs=None,
            ssl_check_hostname=False,
            socket_timeout=60,
            socket_connect_timeout=30,
            decode_responses=False  # Important for binary data
        )
        
        # Test the connection
        connection.ping()
        logger.info("Successfully connected to Upstash Redis using direct parameters")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to Upstash Redis: {e}")
        raise

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

def process_job(job_data):
    """
    Process a job from the queue.
    
    Args:
        job_data (dict or list): A dictionary containing job details or a list where the first item is the job data:
            - version: The data.world version to use
            - input_text: The text to process
            - job_id: A unique identifier for this job
    
    Returns:
        dict: A dictionary with the job results and status
    """
    # Log the received job data type for debugging
    logger.info(f"process_job called with argument type: {type(job_data)}")
    
    # Handle the case where job_data might be wrapped in a list
    if isinstance(job_data, list) and len(job_data) > 0:
        logger.info("Job data was passed as a list, extracting first element")
        job_data = job_data[0]
        
    # Log the processed job data
    logger.info(f"Processing job data: {type(job_data)}, keys: {job_data.keys() if isinstance(job_data, dict) else 'not a dict'}")
    
    try:
        # Initialize job info
        version = job_data.get('version')
        input_text = job_data.get('input_text', '')
        renew = job_data.get('renew', True)
        job_id = job_data.get('job_id', 'unknown')
        
        logger.info(f"Job details: id={job_id}, version={version}, text_length={len(input_text)}")
        
        # Initialize progress reporter
        progress = ProgressReporter(job_id)
        progress.start_stage("initialization", "Initializing job", 0)
        
        logger.info(f"Starting job {job_id}")
        logger.info(f"Version: {version}")
        logger.info(f"Input text length: {len(input_text)}")
        
        # If renew flag is set, clean up old files
        if renew:
            progress.update_stage("initialization", 5, "Cleaning up old files")
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
                        progress.update_stage("initialization", 10, f"Error: Failed to clean up files - {str(e)}")
                        return {
                            "status": "error",
                            "error_type": "file_system",
                            "message": f"Failed to clean up files: {str(e)}",
                            "job_id": job_id
                        }
                logger.info("Cleanup complete")
                progress.update_stage("initialization", 10, "File cleanup complete")
        
        # Load ontology
        progress.start_stage("ontology_loading", "Loading ontology", 10)
        try:
            # Define a function for downloading ontology that can be retried
            @retry_on_exception(max_retries=3, exceptions=[requests.RequestException], backoff_factor=1.5)
            def download_ontology(version_id, token):
                """Download ontology with retry logic"""
                headers = {"Authorization": f"Bearer {token}"}
                url = f"https://api.data.world/v0/file_download/{version_id}/ontology.ttl"
                logger.info(f"Downloading ontology from {url}")
                progress.update_stage("ontology_loading", 15, f"Downloading ontology from data.world")
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                return response.text
            
            # Download data from the data.world API
            if version:
                api_token = os.getenv('DW_AUTH_TOKEN')
                if not api_token:
                    logger.error("DW_AUTH_TOKEN not found in environment variables")
                    progress.update_stage("ontology_loading", 15, "Error: Missing authentication token")
                    return {
                        "status": "error",
                        "error_type": "configuration",
                        "message": "DW_AUTH_TOKEN not found in environment variables",
                        "job_id": job_id
                    }
                    
                # Download with retry
                try:
                    ontology_data = download_ontology(version, api_token)
                    progress.update_stage("ontology_loading", 20, "Ontology downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download ontology after retries: {e}")
                    progress.update_stage("ontology_loading", 20, f"Error: Failed to download ontology - {str(e)}")
                    return {
                        "status": "error",
                        "error_type": "download",
                        "message": f"Failed to download ontology after multiple attempts: {str(e)}",
                        "job_id": job_id
                    }
                    
                # Load the ontology
                progress.update_stage("ontology_loading", 25, "Parsing ontology data")
                load_ontology(ontology_data)
            else:
                # If no version specified, read from local ontology.ttl file
                try:
                    progress.update_stage("ontology_loading", 15, "Reading local ontology file")
                    with open('ontology.ttl', 'r') as file:
                        ontology_data = file.read()
                    progress.update_stage("ontology_loading", 25, "Parsing ontology data")
                    load_ontology(ontology_data)
                except FileNotFoundError:
                    logger.error("Local ontology.ttl file not found")
                    progress.update_stage("ontology_loading", 15, "Error: Local ontology file not found")
                    return {
                        "status": "error",
                        "error_type": "file_not_found",
                        "message": "Local ontology.ttl file not found",
                        "job_id": job_id
                    }
                    
            logger.info("Loaded ontology")
            logger.info(f"Found prefixes: {prefixes}")
            logger.info(f"Found entity types: {entity_types}")
            progress.complete_stage("ontology_loading", "Ontology loaded successfully", 30)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load ontology: {e}")
            progress.update_stage("ontology_loading", 25, f"Error: Failed to parse ontology - {str(e)}")
            return {
                "status": "error",
                "error_type": "parsing",
                "message": f"Failed to load ontology: {str(e)}",
                "job_id": job_id
            }
        except Exception as e:
            logger.error(f"Unhandled error during ontology loading: {e}")
            progress.update_stage("ontology_loading", 25, f"Error: Unhandled error - {str(e)}")
            return {
                "status": "error", 
                "error_type": "unknown",
                "message": f"Unhandled error during ontology loading: {str(e)}",
                "job_id": job_id
            }
        
        # Create resolver with context
        progress.start_stage("graph_initialization", "Initializing graph processing", 30)
        resolver = make_curie_resolver(prefixes, curie_map)
        
        # Create GraphRAG instance
        try:
            progress.update_stage("graph_initialization", 35, "Creating GraphRAG instance")
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
            progress.complete_stage("graph_initialization", "Graph initialized successfully", 40)
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG: {e}", exc_info=True)
            progress.update_stage("graph_initialization", 35, f"Error: Failed to initialize graph - {str(e)}")
            return {
                "status": "error",
                "error_type": "initialization",
                "message": f"Failed to initialize GraphRAG: {str(e)}",
                "job_id": job_id
            }
        
        # Process content and create graph
        progress.start_stage("content_processing", "Processing content", 40)
        try:
            if not input_text:
                logger.error("Missing input_text field in job data")
                progress.update_stage("content_processing", 40, "Error: Missing input text")
                return {
                    "status": "error",
                    "error_type": "missing_input",
                    "message": "Missing input_text field in job data",
                    "job_id": job_id
                }
                
            # Use the input text from the request
            logger.info(f"Using provided input text ({len(input_text)} characters)")
            content = input_text
            
            logger.info("Starting content insertion...")
            progress.update_stage("content_processing", 45, "Processing text content")
            grag.insert(content)
            logger.info("Completed content insertion")
            progress.complete_stage("content_processing", "Content processed successfully", 60)
        except Exception as e:
            logger.error(f"Failed to process content: {e}", exc_info=True)
            progress.update_stage("content_processing", 45, f"Error: Failed to process content - {str(e)}")
            return {
                "status": "error",
                "error_type": "processing",
                "message": f"Failed to process content: {str(e)}",
                "job_id": job_id
            }
        
        # Build and save RDF graph
        progress.start_stage("graph_building", "Building RDF graph", 60)
        try:
            # Run the async function to build the graph
            progress.update_stage("graph_building", 65, "Building graph from extracted data")
            g = get_event_loop().run_until_complete(build_rdf_graph(grag))
            
            # Apply hygiene rules and save RDF graph
            progress.update_stage("graph_building", 75, "Applying hygiene rules to graph")
            g = apply_hygiene_rules(g, ontology)
            
            # Extract version name for the filename (part after the slash)
            if version and '/' in version:
                version_part = version.split('/')[-1]
                filename = f"{version_part}.ontology.ttl"
            else:
                filename = "unknown-version.ontology.ttl"
            
            # Save a local copy for inspection
            progress.update_stage("graph_building", 85, "Saving output files")
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            local_output_path = os.path.join(output_dir, filename)
            g.serialize(destination=local_output_path, format='turtle')
            logger.info(f"Saved local RDF output to {local_output_path}")
            progress.complete_stage("graph_building", "Graph built and saved successfully", 90)
            
            # Upload to data.world if specified
            progress.start_stage("uploading", "Preparing for upload", 90)
            
            # Serialize the graph to a BytesIO object in turtle format for upload
            data_in_memory = BytesIO()
            g.serialize(destination=data_in_memory, format='turtle')
            data_in_memory.seek(0)  # Necessary to rewind the BytesIO object
            
            # Upload to data.world if a version is specified
            upload_status = "Not uploaded (no version specified)"
            if version:
                # Define upload function with retry
                @retry_on_exception(max_retries=3, exceptions=[requests.RequestException], backoff_factor=1.5)
                def upload_to_dataworld(version_id, token, file_data):
                    """Upload file to data.world with retry logic"""
                    filename = "ontology.ttl"
                    url = f"https://api.data.world/v0/uploads/{version_id}/files/{filename}"
                    
                    headers = {
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}"
                    }
                    
                    logger.info(f"Uploading to data.world URL: {url}")
                    progress.update_stage("uploading", 95, "Uploading to data.world")
                    
                    response = requests.put(
                        url,
                        headers=headers,
                        data=file_data,
                        timeout=120
                    )
                    
                    response.raise_for_status()
                    return response
                
                # Your DW_AUTH_TOKEN should be available throughout the execution
                api_token = os.getenv('DW_AUTH_TOKEN')
                if not api_token:
                    logger.error("DW_AUTH_TOKEN not found for upload")
                    progress.update_stage("uploading", 95, "Error: Missing authentication token for upload")
                    return {
                        "status": "error",
                        "error_type": "configuration",
                        "message": "DW_AUTH_TOKEN not found for upload",
                        "job_id": job_id,
                        "local_output": local_output_path
                    }
                
                # Attempt upload with retry
                try:
                    response = upload_to_dataworld(version, api_token, data_in_memory)
                    
                    if response.status_code == 200:
                        logger.info("Successfully uploaded RDF to data.world")
                        upload_status = "Successfully uploaded to data.world"
                        progress.complete_stage("uploading", "Upload complete", 100)
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
                        progress.update_stage("uploading", 95, f"Error: Upload failed - {error_message}")
                        return {
                            "status": "error",
                            "error_type": "upload",
                            "message": f"Failed to upload to data.world: {error_message}",
                            "job_id": job_id,
                            "local_output": local_output_path
                        }
                except Exception as e:
                    logger.error(f"Error during upload to data.world: {e}")
                    progress.update_stage("uploading", 95, f"Error: Upload failed after retries - {str(e)}")
                    return {
                        "status": "error",
                        "error_type": "upload",
                        "message": f"Error during upload: {str(e)}",
                        "job_id": job_id,
                        "local_output": local_output_path
                    }
            else:
                logger.info("No version specified, skipping data.world upload")
                progress.complete_stage("uploading", "No upload required", 100)
            
            # Return success response
            # Make sure to include the version for web app to use
            return {
                "status": "success",
                "message": upload_status,
                "job_id": job_id,
                "local_output": local_output_path,
                "uploaded_to_dataworld": bool(version),
                "version": version  # Explicitly include version
            }
        except Exception as e:
            logger.error(f"Error in graph processing: {e}", exc_info=True)
            current_stage = progress.get_job_progress(job_id).get('stage', 'unknown')
            progress.update_stage(current_stage, 80, f"Error: Graph processing failed - {str(e)}")
            return {
                "status": "error",
                "error_type": "processing",
                "message": f"Failed to build RDF graph: {str(e)}",
                "job_id": job_id
            }
    
    except Exception as e:
        logger.error(f"Unhandled exception in worker process: {e}", exc_info=True)
        # Try to update progress even though we hit an unhandled exception
        try:
            progress = ProgressReporter(job_data.get('job_id', 'unknown'))
            progress.update_progress(0, "error", f"Unhandled exception: {str(e)}")
        except:
            pass
            
        return {
            "status": "error",
            "error_type": "unhandled",
            "message": f"Unhandled exception: {str(e)}",
            "job_id": job_data.get('job_id', 'unknown')
        }

def main():
    try:
        # Connect to Redis
        redis_conn = get_redis_connection()
        
        # Test Redis connection and key operations
        logger.info("Testing Redis connection and operations...")
        test_key = "worker_test_key"
        test_value = f"test_value_{datetime.datetime.now().isoformat()}"
        
        # Try to set and get a test key
        try:
            redis_conn.set(test_key, test_value)
            retrieved = redis_conn.get(test_key)
            if retrieved:
                retrieved_value = retrieved.decode('utf-8') if isinstance(retrieved, bytes) else retrieved
                logger.info(f"Redis test: SET/GET successful. Stored '{test_value}', retrieved '{retrieved_value}'")
                
                # Test hash operations that are used for job metadata
                hash_key = "worker_test_hash"
                hash_data = {
                    "field1": "value1",
                    "field2": "value2",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                redis_conn.hmset(hash_key, hash_data)
                hash_result = redis_conn.hgetall(hash_key)
                
                if hash_result:
                    logger.info(f"Redis hash operations successful. Hash entries: {len(hash_result)}")
                else:
                    logger.warning("Redis hash operations failed: No data retrieved")
            else:
                logger.warning("Redis test: SET successful but GET returned None")
        except Exception as e:
            logger.error(f"Redis test operations failed: {e}")
        
        # Start the worker
        logger.info("Starting worker process...")
        
        # Ensure queue name is consistent with web app
        queue_name = 'graphrag_processing'
        logger.info(f"Initializing worker to monitor queue: {queue_name}")
        
        # Create a worker
        worker = Worker([queue_name], connection=redis_conn)
        logger.info(f"Worker created for queue '{queue_name}' and ready to process jobs")
        
        # Log RQ queue details
        logger.info(f"Registered worker queues: {worker.queues}")
        try:
            queue = Queue(queue_name, connection=redis_conn)
            # Get all job IDs in the queue
            queue_jobs = queue.job_ids
            logger.info(f"Queue status: '{queue.name}', jobs count: {len(queue_jobs)}")
            logger.info(f"Jobs in queue: {queue_jobs}")
            
            # Check scheduled and finished jobs
            scheduled_jobs = queue.scheduled_job_registry.get_job_ids()
            logger.info(f"Scheduled jobs: {scheduled_jobs}")
            finished_jobs = queue.finished_job_registry.get_job_ids()
            logger.info(f"Finished jobs: {finished_jobs}")
            failed_jobs = queue.failed_job_registry.get_job_ids()
            logger.info(f"Failed jobs: {failed_jobs}")
        except Exception as e:
            logger.error(f"Failed to get queue information: {e}")
        
        # Start working (processing jobs)
        # Note: Using only parameters known to be supported in the installed RQ version
        logger.info("Worker starting to process jobs")
        worker.work(with_scheduler=True)
    except Exception as e:
        logger.error(f"Error in worker main function: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()