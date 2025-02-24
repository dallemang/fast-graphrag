"""Example usage of GraphRAG with Gemini LLM and Embeddings from VertexAI"""
# DEPENDENCIES: pip install fast-graphrag google-cloud-aiplatform
# VertexAI: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/try-gen-ai
# Login with your Google Cloud account in CLI: gcloud auth application-default login
# Run this example: python gemini_vertexai_llm.py

import asyncio
import re
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast, Literal, TypeVar, Callable
from functools import wraps
import logging
import instructor

import numpy as np
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel

from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService
from fast_graphrag._models import BaseModelAlias
from fast_graphrag._models import _json_schema_slim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

T_model = TypeVar("T_model")

def throttle_async_func_call(
    max_concurrent: int = 2048, stagger_time: Optional[float] = None, waiting_time: float = 0.001
):
    _wrappedFn = TypeVar("_wrappedFn", bound=Callable[..., Any])

    def decorator(func: _wrappedFn) -> _wrappedFn:
        __current_exes = 0
        __current_queued = 0

        @wraps(func)
        async def wait_func(*args: Any, **kwargs: Any) -> Any:
            nonlocal __current_exes, __current_queued
            while __current_exes >= max_concurrent:
                await asyncio.sleep(waiting_time)

            # __current_queued += 1
            # await asyncio.sleep(stagger_time * (__current_queued - 1))
            # __current_queued -= 1
            __current_exes += 1
            result = await func(*args, **kwargs)
            __current_exes -= 1
            return result

        return wait_func  # type: ignore

    return decorator

class LLMServiceNoResponseError(Exception):
    """Raised when the LLM service returns no response."""
    pass


@dataclass
class VertexAIEmbeddingService(BaseEmbeddingService):
    """Vertex AI implementation for text embeddings using the Gecko model."""
    # Custom configuration for Vertex AI's Gecko embedding model
    embedding_dim: int = field(default=768)  # Gecko model dimension
    max_elements_per_request: int = field(default=32)  # Match OpenAI's batch size for compatibility
    model: Optional[str] = field(default="textembedding-gecko@latest")
    client: Literal["vertex"] = field(default="vertex")

    def __post_init__(self):
        """Initialize the Vertex AI embedding model."""
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.model)
        logger.debug("Initialized VertexAIEmbeddingService with Vertex AI client.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Generates embeddings for a list of texts using batched processing.
        
        Args:
            texts: List of strings to embed
            model: Optional model override
        Returns:
            Numpy array of embeddings
        """
        # Custom batching implementation to match OpenAI's behavior
        batched_texts = [
            texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
            for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
        ]
        
        # Process batches concurrently
        response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])
        
        # Flatten the batched responses and convert to numpy array
        embeddings = np.vstack([batch_embeddings for batch_embeddings in response])
        logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TimeoutError, Exception)),  
    )
    async def _embedding_request(self, input_texts: List[str], model: str) -> np.ndarray:
        """Get embeddings for a batch of texts.

        Args:
            input_texts (List[str]): Batch of texts to embed
            model (str): Model name

        Returns:
            np.ndarray: Array of embeddings for the batch
        """
        try:
            # Get embeddings for the batch
            embeddings = await self._embedding_model.get_embeddings_async(input_texts)
            
            # Convert to numpy array maintaining the same format as OpenAI
            return np.array([emb.values for emb in embeddings], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error in embedding request: {str(e)}")
            raise


@dataclass
class VertexAILLMService(BaseLLMService):
    """Vertex AI implementation for LLM services using Gemini models."""
    model: Optional[str] = field(default="gemini-1.0-pro")
    base_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    
    # Other fields with defaults
    max_retries: int = field(default=3)
    retry_delay: float = field(default=1.0)
    rate_limit_max_retries: int = field(default=5)
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    temperature: float = field(default=0.6)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    llm_calls_count: int = field(default=0, init=False)
    _vertex_model: Any = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize after dataclass initialization."""
        if self.model is None:
            raise ValueError("Model name must be provided.")
            
        # Remove event loop handling and just initialize the Vertex AI model
        self._vertex_model = GenerativeModel(self.model)

    def _extract_retry_time(self, error_message: str) -> float:
        # Custom retry time extraction from Vertex AI error messages
        match = re.search(r'Retry the request after (\d+) sec', str(error_message))
        if match:
            return float(match.group(1))
        return 2.0

    def _count_tokens(self, messages: List[dict[str, str]]) -> int:
        return sum(len(msg["content"].split()) * 1.3 for msg in messages)
    

    @throttle_async_func_call(
        max_concurrent=50,
        stagger_time=0.1,
        waiting_time=0.001
    )
    async def send_message(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict[str, str]]] = None,
        response_model: Optional[Type[T_model]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[T_model, List[dict[str, str]]]:
        """Sends a message to the Vertex AI LLM and handles the response.

        Args:
            prompt: Main input text
            model: Optional model override
            system_prompt: Optional system instructions
            history_messages: Previous conversation messages
            response_model: Expected response structure
            temperature: Optional temperature override
            **kwargs: Additional parameters

        Returns:
            Tuple of (parsed response, message history)
        
        Raises:
            LLMServiceNoResponseError: If no valid response is received
            ValueError: If model name is missing
        """
        def convert_to_vertex_schema(schema_to_convert):
            """Converts JSON Schema to Vertex AI's expected schema format.
            
            Args:
                schema_to_convert: JSON Schema to convert
            Returns:
                Dict containing Vertex AI compatible schema
            """
            if "$ref" in schema_to_convert:
                ref_path = schema_to_convert["$ref"].split("/")
                ref_schema = schema
                for path in ref_path[1:]:
                    ref_schema = ref_schema[path]
                return convert_to_vertex_schema(ref_schema)
            
            if schema_to_convert["type"] == "array":
                return {
                    "type": "array",
                    "items": convert_to_vertex_schema(schema_to_convert["items"])
                }
            elif schema_to_convert["type"] == "object":
                props = {}
                for prop_name, prop_schema in schema_to_convert.get("properties", {}).items():
                    props[prop_name] = convert_to_vertex_schema(prop_schema)
                return {
                    "type": "object",
                    "properties": props
                }
            else:
                return {
                    "type": schema_to_convert["type"],
                    "description": schema_to_convert.get("description", "")
                }


        temperature = self.temperature
        retries = 0
        rate_limit_retries = 0
        
        logger.debug(f"Sending message with prompt: {prompt}")
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")
        
        messages: List[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"Added system prompt: {system_prompt}")

        if history_messages:
            messages.extend(history_messages)
            logger.debug(f"Added history messages: {history_messages}")

        messages.append({"role": "user", "content": prompt})

        # Add format instruction to the prompt if response_model exists
        if response_model:
            model_class = (response_model.Model 
                        if issubclass(response_model, BaseModelAlias)
                        else response_model)
            schema = model_class.model_json_schema()

            # Convert JSON Schema to Vertex AI Schema format
            vertex_schema = {
                "type": "object",
                "properties": {}
            }
            
            # Convert each top-level property
            for prop_name, prop_schema in schema["properties"].items():
                vertex_schema["properties"][prop_name] = convert_to_vertex_schema(prop_schema)


        # Custom schema instruction for ensuring consistent JSON responses, until Instructor fully supports VertexAI
        schema_instruction = (
            "IMPORTANT: Your response must be a valid JSON object containing all the following fields "
            "(use empty arrays [] for fields with no values):\n"
            "- entities\n"
            "- relationships\n"
            "- other_relationships\n\n"
            "Each entity MUST contain ALL of these required fields:\n"
            "- name: The unique identifier of the entity\n"
            "- type: The category or classification of the entity\n"
            "- desc: A detailed description of the entity (REQUIRED, never omit this field)\n\n"
            "Example of a valid entity:\n"
            "{\n"
            '    "name": "Scrooge",\n'
            '    "type": "person",\n'
            '    "desc": "A miserly businessman who undergoes a transformation"\n'
            "}\n\n"
            "Example of a complete valid response format:\n"
            "{\n"
            '    "entities": [\n'
            '        {"name": "Scrooge", "type": "person", "desc": "A miserly businessman"},\n'
            '        {"name": "London", "type": "location", "desc": "The city where the story takes place"}\n'
            '    ],\n'
            '    "relationships": [\n'
            '        {"source": "Scrooge", "target": "London", "desc": "Scrooge lives and works in London"}\n'
            '    ],\n'
            '    "other_relationships": []\n'
            "}"
        )

        # Insert instruction at the beginning of messages until Instructor fully supports VertexAI
        messages.insert(0, {
            "role": "system",
            "content": schema_instruction
        })

        
        # Combine messages into a single prompt
        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        while True:
            try:

                # Custom configuration for Vertex AI generation parameters
                generation_config = GenerationConfig(
                    temperature=temperature,
                    candidate_count=1,
                    response_mime_type="application/json",
                    response_schema=vertex_schema  # Use the converted schema
                )

                # Use async version of generate_content
                vertex_response = await self._vertex_model.generate_content_async(
                    combined_prompt,
                    generation_config=generation_config,
                    stream=False
                )

                # Validate response
                if not vertex_response or not vertex_response.text:
                    logger.error("Empty response from Vertex AI")
                    raise LLMServiceNoResponseError("Empty response from Vertex AI")

                # Extract text content from response
                response_text = vertex_response.text

                # Parse response using response_model if provided
                try:
                    if response_model:
                        if issubclass(response_model, BaseModelAlias):
                            llm_response = response_model.Model.model_validate_json(response_text)
                        else:
                            llm_response = response_model.model_validate_json(response_text)
                    else:
                        llm_response = response_text
                except ValidationError as e:
                    logger.error(f"JSON validation error: {str(e)}")
                    raise LLMServiceNoResponseError(f"Invalid JSON response: {str(e)}") from e


                self.llm_calls_count += 1

                original_llm_response = llm_response

                if not llm_response:
                    logger.error("No response received from the language model.")
                    raise LLMServiceNoResponseError("No response received from the language model.")

                messages.append({
                    "role": "assistant",
                    "content": (llm_response.model_dump_json() 
                              if isinstance(llm_response, BaseModel) 
                              else str(llm_response)),
                })
                logger.debug(f"Received response: {llm_response}")

                if response_model and issubclass(response_model, BaseModelAlias):
                    llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

                return llm_response, messages

            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
                token_count = round(self._count_tokens(messages))
                
                if status_code == 500:
                    if retries >= self.max_retries:
                        error_log = (
                            f"{timestamp}|500|{model}|{token_count}|"
                            f"Max retries reached ({self.max_retries})|{str(e)}\n"
                        )
                        print(error_log)
                        err = f"LLM API failed with 500 error after {self.max_retries} retries: {e}"
                        logger.error(err)
                        raise Exception(err) from e

                    retries += 1
                    wait_time = self.retry_delay * (retries)
                    error_log = f"{timestamp}|500|{model}|{token_count}|Attempt {retries}|{str(e)}\n"
                    print(error_log)
    
                    await asyncio.sleep(wait_time)
                    continue

                if status_code == 429:
                    if rate_limit_retries >= self.rate_limit_max_retries:
                        error_log = (
                            f"{timestamp}|429|{model}|{token_count}|"
                            f"Rate limit max retries reached ({self.rate_limit_max_retries})|{str(e)}\n"
                        )
                        print(error_log)
                        raise Exception(f"Rate limit exceeded after {self.rate_limit_max_retries} retries: {e}") from e
                    
                    retry_time = self._extract_retry_time(str(e))
                    rate_limit_retries += 1
                    error_log = f"{timestamp}|429|{model}|{token_count}|Attempt {rate_limit_retries}|{str(e)}\n"
                    print(error_log)
                    err = (f"Rate limit hit (attempt {rate_limit_retries}/{self.rate_limit_max_retries}). "
                          f"Waiting {retry_time} seconds before retry...")
                    
                    logger.warning(err)
                    await asyncio.sleep(retry_time)
                    continue
                raise

            except APIConnectionError as e:
                raise

            except Exception as e:
                err = f"Unexpected error: {str(e)}"
                logger.error(err)
                raise LLMServiceNoResponseError(err) from e


if __name__ == "__main__":
    from fast_graphrag import GraphRAG

    DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

    EXAMPLE_QUERIES = [
        "What is the significance of Christmas Eve in A Christmas Carol?",
        "How does the setting of Victorian London contribute to the story's themes?",
        "Describe the chain of events that leads to Scrooge's transformation.",
        "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
        "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
    ]

    # Custom entity types for story analysis
    ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

    # Initialize Vertex AI
    vertexai.init(#project="<your-project-id>", 
                  #location="<your-region>"
                    )     
    # Initialize both LLM and embedding services
    embedding_service = VertexAIEmbeddingService()
    
    llm_service = VertexAILLMService(
        model="gemini-1.5-pro-002",
        max_retries=2,
        retry_delay=1.0,
        rate_limit_max_retries=2,
        temperature=0.6
    )

    # Initialize GraphRAG with VertexAI services
    grag = GraphRAG(
        working_dir="/tmp/book_example",
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service 
        )
    )

    # Read and process the book
    with open("../mock_data.txt") as f:
        grag.insert(f.read())

    # Run an example query
    user_query = "Who are three main characters in the story?"
    print(f"User query: {user_query}")
    print(grag.query(user_query).response)

