"""Prompts."""

from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

## NEW
PROMPTS["entity_relationship_extraction"] = """# DOMAIN PROMPT

Here is the ontology in OWL (TTL) format:
{domain}

# GOAL
Your goal is to highlight information that is relevant to the ontology and the questions that may be asked on it.
Given an input document, identify all relevant entities and all relationships among them. 
For each entity, you must find the class in the ontology that best matches the entity.  Do not make up your own classes, and do not use "UNKNOWN" or "OTHER".

Examples of possible questions:
{example_queries}

# STEPS
1. Identify all entities that are members of classes in the domain description. Make sure to extract all and only the entities that are of one of the classes defined in the ontology. Use singular names and split compound concepts when necessary (for example, from the sentence "they are movie and theater directors", you should extract the entities "movie director" and "theater director").
2. Identify all relationships between the entities found in step 1. Clearly resolve pronouns to their specific names to maintain clarity. Use properties from the ontology whenever you can.  
3. If you find relationships that are not defined in the ontology, provide a description of the relationship.
4. Double check that each entity identified in step 1 appears in at least one relationship. If not, add the missing relationships.


# EXAMPLE DATA
Example ontology: ex:Location a owl:Class. ex:Organization a owl:Class. ex:Person a owl:Class. ex:Communication a owl:Class. 
                  ex:locatedIn a owl:ObjectProperty. ex:broadcastsIn a owl:ObjectProperty. ex:speaks a owl:ObjectProperty. ex:offers a owl:ObjectProperty. ex:launched a owl:ObjectProperty.
Example document: Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into new media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."





Output:
{{
"entities": [
	{{"name": "RADIO CITY", "rdf:type": "ex:organization", "desc": "Radio City is India's first private FM radio station"}},
	{{"name": "INDIA", "rdf:type": "ex:location", "desc": "A country"}},
	{{"name": "FM RADIO STATION", "rdf:type": "ex:communication", "desc": "A radio station that broadcasts using frequency modulation"}},

	{{"name": "ENGLISH", "type": "communication", "desc": "A language"}},
	{{"name": "HINDI", "type": "communication", "desc": "A language"}},
	{{"name": "NEW MEDIA", "type": "communication", "desc": "New media"}},
	{{"name": "PLANETRADIOCITY", "type": "organization", "desc": "PlanetRadiocity.com is an online music portal"}},
	{{"name": "MUSIC PORTAL", "subclassOf": "ex:Channel", "desc": "A website that offers music related information"}},
	{{"name": "NEWS", "subclassOf": "ex:communication", "desc": "News"}},
	{{"name": "VIDEO", "subclassOf": "ex:communication", "desc": "Video"}},
	{{"name": "SONG", "subclassOf": "ex:communication", "desc": "Song"}}

],
"relationships": [
	{{"source": "RADIO CITY", "target": "INDIA", "desc": "Radio City is located in India", "property": "ex:locatedIn"}},
	{{"source": "RADIO CITY", "target": "FM RADIO STATION", "desc": "Radio City is a private FM radio station started on 3 July 2001", "property": "ex:type"}},

	{{"source": "RADIO CITY", "target": "ENGLISH", "desc": "Radio City broadcasts English songs", "property": "ex:broadcastsIn"}},


	{{"source": "RADIO CITY", "target": "HINDI", "desc": "Radio City broadcasts songs in the Hindi language", "property": "ex:broadcastsIn"}},

	{{"source": "RADIO CITY", "target": "PLANETRADIOCITY", "desc": "Radio City launched PlanetRadiocity.com in May 2008", "property": "ex:launched"}},
	{{"source": "PLANETRADIOCITY", "target": "MUSIC PORTAL", "desc": "PlanetRadiocity.com is a music portal", "property": "rdf:type"}},




	{{"source": "PLANETRADIOCITY", "target": "NEWS", "desc": "PlanetRadiocity.com offers music related news", "property": "ex:offers"}},
	{{"source": "PLANETRADIOCITY", "target": "SONG", "desc": "PlanetRadiocity.com offers songs", "property": "ex:offers"}},
	{{"source": "PLANETRADIOCITY", "target": "VIDEO", "desc": "PlanetRadiocity.com offers music related videos", "property": "ex:offers"}}




],
"other_relationships": [
	{{"source": "RADIO CITY", "target": "NEW MEDIA", "desc": "Radio City forayed into new media in May 2008.", "property": "ex:newProperty"}},

]
}}

# INPUT DATA

Document: {input_text}

Output:
"""

PROMPTS["entity_relationship_continue_extraction"] = "MANY entities were missed in the last extraction.  Add them below using the same format:"

PROMPTS["entity_relationship_gleaning_done_extraction"] = "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added."

PROMPTS["entity_extraction_query"] = """Given the query below, your task is to extract all entities relevant to perform information retrieval to produce an answer.

-EXAMPLE 1-
Query: Who directed the film that was shot in or around Leland, North Carolina in 1986?
Ouput: {{"named": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"], "generic": ["film director"]}}

-EXAMPLE 2-
Query: What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?
Ouput: {{"named": ["[BASEBALL PLAYER] Fred Gehrke", "[EVENT] 2010 Major League Baseball Draft"], "generic": ["23rd baseball draft pick"]}}

-INPUT-
Query: {query}
Output:
"""


PROMPTS[
	"summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a summary of the data provided below.
Given the current description, summarize it by removing redundant and generic information. Resolve any contradictions and provide a single, coherent summary.
Write in third person and explicitly include the entity names to preserve the full context.

Current:
{description}

Updated:
"""


PROMPTS[
	"edges_group_similar"
] = """You are a helpful assistant responsible for maintaining a list of facts describing the relations between two entities so that information is not redundant.
Given a list of ids and facts, identify any facts that should be grouped together as they contain similar or duplicated information and provide a new summarized description for the group.

# EXAMPLE
Facts (id, description):
0, Mark is the dad of Luke
1, Luke loves Mark
2, Mark is always ready to help Luke
3, Mark is the father of Luke
4, Mark loves Luke very much

Ouput:
{{
	grouped_facts: [
	{{
		'ids': [0, 3],
		'description': 'Mark is the father of Luke'
	}},
	{{
		'ids': [1, 4],
		'description': 'Mark and Luke love each other very much'
	}}
	]
}}

# INPUT:
Facts:
{edge_list}

Ouput:
"""

PROMPTS["generate_response_query_with_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. While you write the response, you must include inline references to the all the sources you are using by appending `[<source_id>]` at the end of each sentence, where `source_id` is the corresponding source ID from the "Sources" list.
5. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["generate_response_query_no_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
