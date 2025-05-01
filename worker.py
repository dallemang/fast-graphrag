import os
import logging
import traceback
import datetime
from io import BytesIO
import requests
from rq import get_current_job
from rdflib import Graph, URIRef, Literal, Dataset, Namespace
from rdflib.namespace import RDF, OWL, RDFS, SKOS, XSD
from urllib.parse import urlparse
from typing import Dict, Optional
import shutil

from fast_graphrag import GraphRAG
from fast_graphrag._utils import get_event_loop

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System namespace for GraphRAG
SE = Namespace("https://models.data.world/graphrag/")  # type: ignore

# Use Redis to store job results
from redis import Redis
import json
import os

# Connect to Redis with a special approach for Heroku
# Critical fix for "Connection reset by peer" errors
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
logger.info(f"Using REDIS_URL: {redis_url}")

# Special handling for Heroku Redis
is_heroku_redis = "compute-1.amazonaws.com" in redis_url
if is_heroku_redis:
    logger.info("Detected Heroku Redis URL - applying special handling")
    # Extract credentials and host
    import re
    from urllib.parse import urlparse, parse_qs
    
    # Parse the URL to work with it
    parsed_url = urlparse(redis_url)
    
    # Check if we already have query parameters
    query_params = parse_qs(parsed_url.query)
    
    # Don't override existing parameters if they exist
    connection_params = {
        'socket_timeout': int(query_params.get('socket_timeout', [60])[0]),
        'socket_connect_timeout': int(query_params.get('socket_connect_timeout', [30])[0]),
        'socket_keepalive': query_params.get('socket_keepalive', ['true'])[0] == 'true',
        'retry_on_timeout': True,
        'health_check_interval': int(query_params.get('health_check_interval', [15])[0]),
        'decode_responses': False,  # Important for binary data
    }
    
    logger.info(f"Using special Heroku Redis connection parameters: {connection_params}")

# First try: Full-featured connection with all necessary parameters
try:
    if is_heroku_redis:
        redis_conn = Redis.from_url(redis_url, **connection_params)
    else:
        redis_conn = Redis.from_url(
            redis_url,
            socket_timeout=60,
            socket_connect_timeout=30,
            retry_on_timeout=True,
            decode_responses=False,  # Important for binary data
            health_check_interval=30  # Regular health checks to detect disconnections
        )
    # Test connection with longer timeout for Heroku
    redis_conn.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    
    # Second try: Simplified connection with just essential parameters
    try:
        logger.info("Trying simplified connection with minimal options")
        if is_heroku_redis:
            # For Heroku, just use very basic parameters focused on preventing resets
            redis_conn = Redis.from_url(
                redis_url,
                socket_timeout=90,  # Extra long timeout
                retry_on_timeout=True
            )
        else:
            redis_conn = Redis.from_url(redis_url)
            
        redis_conn.ping()
        logger.info("Simplified Redis connection successful")
    except Exception as e:
        logger.error(f"Simplified Redis connection also failed: {e}")
        
        # Last resort: Use a mock Redis implementation that doesn't fail but doesn't do anything
        logger.critical("ALL Redis connection attempts failed! Creating mock Redis for critical functionality")
        
        # Create a minimal mock Redis that won't fail but won't do much either
        class MockRedis:
            def __init__(self):
                logger.warning("Using MockRedis - only minimal functionality will be available")
            
            def ping(self):
                return True
                
            def set(self, key, value):
                logger.info(f"MockRedis SET: {key}")
                return True
                
            def get(self, key):
                logger.info(f"MockRedis GET: {key}")
                return None
        
        redis_conn = MockRedis()

# Key prefix for job results
JOB_PREFIX = 'graphrag:job:'

def get_local_part(url):
    parsed_url = urlparse(url)
    local_part = parsed_url.fragment if parsed_url.fragment else parsed_url.path.split('/')[-1]
    return local_part

def load_ontology(ontology_data: str) -> tuple:
    """
    Load ontology from TTL string and return processed data.
    """
    prefixes = {}
    domain = ontology_data 
    entity_types = []
    ontology = Graph()
    curie_map = {}  # Case-insensitive CURIE lookup
    
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
    
    return prefixes, domain, entity_types, ontology, curie_map

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

async def build_rdf_graph(grag):
    try:
        # Process nodes from the graph storage
        graph_storage = grag.state_manager.graph_storage
        chunk_storage = grag.state_manager.chunk_storage
        
        # Create RDF graph
        g = Graph()
        g.bind("gr", SE)  # 'gr' for GraphRAG
        
        # Copy prefixes from ontology
        for prefix, uri in grag.config.prefixes.items():
            g.bind(prefix, uri)
            
        node_count = await graph_storage.node_count()
        logger.info(f"Total nodes in graph: {node_count}")
        
        # Track node URIs for building relationships
        node_uris = {}
        node_names = {}
        
        # Create resolver with context
        resolver = make_curie_resolver(grag.config.prefixes, grag.config.curie_map)
        
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
            
            # Try to resolve type as CURIE first
            logger.info(f"Trying to resolve type for node: {name}, type: {node_type}")
            
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
            elif node_type and node_type.lower() in [et.lower() for et in grag.entity_types]:
                logger.info(f"Node type '{node_type}' found in entity_types")
                
                class_uri = grag.config.curie_map.get(node_type.lower(), None)
                if class_uri is None:
                    logger.warning(f"❌ Node type {node_type} resulted in None for CURIE resolution")
                    # Try to find similar keys
                    similar_keys = [k for k in grag.config.curie_map.keys() if node_type.lower() in k or k in node_type.lower()]
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
                        if not any(grag.config.ontology.triples((resolved, RDF.type, None))):
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

def process_content(job_id, version, input_text, title=None, renew=True):
    """Background worker function to process content and build a graph"""
    try:
        job = get_current_job()
        
        if job:
            job.meta['status'] = 'starting'
            job.meta['progress'] = 0
            job.save_meta()
        
        logger.info(f"Worker starting job {job_id} for version {version}, title {title}")
        result = {
            'status': 'processing',
            'job_id': job_id,
            'version': version,
            'title': title if title else '',
            'start_time': datetime.datetime.now().isoformat(),
            'progress': 0,
            'message': 'Starting processing'
        }
    except Exception as e:
        logger.error(f"Error initializing job {job_id}: {e}", exc_info=True)
        result = {
            'status': 'error',
            'job_id': job_id,
            'version': version,
            'title': title if title else '',
            'message': f"Error initializing job: {str(e)}",
            'traceback': traceback.format_exc()
        }
    
    # Store initial status in Redis
    redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
    
    try:
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
                        result['status'] = 'error'
                        result['message'] = f"Failed to clean up files: {str(e)}"
                        redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                        return result
                logger.info("Cleanup complete")
        
        if job:
            job.meta['status'] = 'loading ontology'
            job.meta['progress'] = 10
            job.save_meta()
            
        result['status'] = 'loading ontology'
        result['progress'] = 10
        result['message'] = 'Loading ontology'
        redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
        # Load ontology
        try:
            # Download data from the data.world API
            if version:
                api_token = os.getenv('DW_AUTH_TOKEN')
                if not api_token:
                    result['status'] = 'error'
                    result['message'] = "DW_AUTH_TOKEN not found in environment variables"
                    redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                    return result
                    
                headers = {"Authorization": f"Bearer {api_token}"}
                url = f"https://api.data.world/v0/file_download/{version}/ontology.ttl"
                logger.info(f"Downloading ontology from {url}")
                response = requests.get(url, headers=headers)
                if not response.ok:
                    response.raise_for_status()
                ontology_data = response.text
                prefixes, domain, entity_types, ontology, curie_map = load_ontology(ontology_data)
            else:
                # If no version specified, read from local ontology.ttl file
                try:
                    with open('ontology.ttl', 'r') as file:
                        ontology_data = file.read()
                        prefixes, domain, entity_types, ontology, curie_map = load_ontology(ontology_data)
                except FileNotFoundError:
                    result['status'] = 'error'
                    result['message'] = "Local ontology.ttl file not found"
                    job_results[job_id] = result
                    return result
                    
            logger.info("Loaded ontology")
            logger.info(f"Found prefixes: {prefixes}")
            logger.info(f"Found entity types: {entity_types}")
            
            if job:
                job.meta['status'] = 'processing content'
                job.meta['progress'] = 20
                job.save_meta()
                
            result['status'] = 'processing content'
            result['progress'] = 20
            result['message'] = 'Ontology loaded, processing content'
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
        except (FileNotFoundError, ValueError) as e:
            result['status'] = 'error'
            result['message'] = f"Failed to load ontology: {str(e)}"
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            return result
        except requests.HTTPError as e:
            result['status'] = 'error'
            result['message'] = f"Failed to download ontology: {str(e)}"
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            return result

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
                resolve_curie=resolver
            )
        )
        
        # Store additional context in GraphRAG object for access in build_rdf_graph
        grag.config.prefixes = prefixes
        grag.config.curie_map = curie_map
        grag.config.ontology = ontology

        # Process content
        try:
            logger.info(f"Using provided input text ({len(input_text)} characters)")
            logger.info("Starting content insertion...")
            grag.insert(input_text)
            logger.info("Completed content insertion")
            
            if job:
                job.meta['status'] = 'building graph'
                job.meta['progress'] = 50
                job.save_meta()
                
            result['status'] = 'building graph'
            result['progress'] = 50
            result['message'] = 'Content processed, building RDF graph'
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"Failed to process content: {str(e)}"
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            return result
        
        # Build and save RDF graph
        try:
            # Run the async function to build the graph
            g = get_event_loop().run_until_complete(build_rdf_graph(grag))
            
            if job:
                job.meta['status'] = 'applying hygiene rules'
                job.meta['progress'] = 70
                job.save_meta()
                
            result['status'] = 'applying hygiene rules'
            result['progress'] = 70
            result['message'] = 'Graph built, applying hygiene rules'
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
            # Apply hygiene rules and save RDF graph
            g = apply_hygiene_rules(g, ontology)
            
            if job:
                job.meta['status'] = 'saving output'
                job.meta['progress'] = 80
                job.save_meta()
                
            result['status'] = 'saving output'
            result['progress'] = 80
            result['message'] = 'Hygiene rules applied, saving output'
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
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
            
            if job:
                job.meta['status'] = 'uploading'
                job.meta['progress'] = 90
                job.save_meta()
                
            result['status'] = 'uploading'
            result['progress'] = 90
            result['message'] = 'Output saved, uploading to data.world'
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
            # Serialize the graph to a BytesIO object in turtle format for upload
            data_in_memory = BytesIO()
            g.serialize(destination=data_in_memory, format='turtle')
            data_in_memory.seek(0)  # Necessary to rewind the BytesIO object

            # Upload to data.world if a version is specified
            if version:
                # Your DW_AUTH_TOKEN should be available throughout the execution
                api_token = os.getenv('DW_AUTH_TOKEN')
                if not api_token:
                    result['status'] = 'error'
                    result['message'] = "DW_AUTH_TOKEN not found for upload"
                    redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                    return result
                    
                file_to_upload = "ontology.ttl"
                url = f"https://api.data.world/v0/uploads/{version}/files/{file_to_upload}"
                
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
                        
                        if job:
                            job.meta['status'] = 'completed'
                            job.meta['progress'] = 100
                            job.save_meta()
                            
                        result['status'] = 'completed'
                        result['progress'] = 100
                        result['message'] = 'Successfully processed and uploaded to data.world'
                        result['local_output'] = local_output_path
                        result['filename'] = filename
                        result['uploaded_to_dataworld'] = True
                        result['complete_time'] = datetime.datetime.now().isoformat()
                        redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                        
                    else: 
                        logger.error(f"Failed to upload to data.world: {response.text}")
                        result['status'] = 'error'
                        result['message'] = f"Failed to upload to data.world: {response.text}"
                        result['local_output'] = local_output_path
                        result['filename'] = filename
                        redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                        return result
                except Exception as e:
                    logger.error(f"Error during upload to data.world: {e}")
                    result['status'] = 'error'
                    result['message'] = f"Error during upload: {str(e)}"
                    result['local_output'] = local_output_path
                    result['filename'] = filename
                    redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
                    return result
            else:
                logger.info("No version specified, skipping data.world upload")
                
                if job:
                    job.meta['status'] = 'completed'
                    job.meta['progress'] = 100
                    job.save_meta()
                    
                result['status'] = 'completed'
                result['progress'] = 100
                result['message'] = 'Successfully processed content'
                result['local_output'] = local_output_path
                result['filename'] = filename
                result['uploaded_to_dataworld'] = False
                result['complete_time'] = datetime.datetime.now().isoformat()
                # Store result in Redis
                redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in graph processing: {e}", exc_info=True)
            result['status'] = 'error'
            result['message'] = f"Failed to build RDF graph: {str(e)}"
            result['traceback'] = traceback.format_exc()
            # Store error result in Redis
            redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
            return result
            
    except Exception as e:
        logger.error(f"Unexpected error in worker: {e}", exc_info=True)
        result['status'] = 'error'
        result['message'] = f"Unexpected error: {str(e)}"
        result['traceback'] = traceback.format_exc()
        # Store error result in Redis
        redis_conn.set(f"{JOB_PREFIX}{job_id}", json.dumps(result))
        return result

def get_job_status(job_id):
    """Get the status of a job by ID"""
    # Handle the case where we're using MockRedis
    if not isinstance(redis_conn, Redis):
        logger.warning(f"Using MockRedis - returning fake job status for {job_id}")
        # Return a mock status that won't crash the app but will show the Redis issue
        return {
            'status': 'error',
            'message': 'Redis connection unavailable - Cannot process jobs',
            'progress': 0,
            'redis_error': 'Connection reset by peer - Redis connection failed',
            'job_id': job_id
        }
    
    # Normal Redis operation
    key = f"{JOB_PREFIX}{job_id}"
    job_data = redis_conn.get(key)
    
    if job_data:
        try:
            return json.loads(job_data)
        except Exception as e:
            logger.error(f"Error deserializing job data: {e}")
            return {'status': 'error', 'message': f'Error retrieving job data: {str(e)}'}
    else:
        # Check if job exists in RQ
        from rq.job import Job
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_queued:
                return {'status': 'queued', 'message': 'Job is queued for processing', 'progress': 0}
            elif job.is_started:
                return {'status': 'processing', 'message': 'Job is currently processing', 'progress': job.meta.get('progress', 10)}
            elif job.is_finished:
                return {'status': 'completed', 'message': 'Job completed but no result found', 'progress': 100}
            elif job.is_failed:
                return {'status': 'error', 'message': f'Job failed: {job.exc_info}', 'progress': 0}
        except Exception as e:
            logger.error(f"Error checking job in RQ: {e}")
            # Not found in RQ either
            pass
            
        return {'status': 'not_found', 'message': f'Job {job_id} not found'}