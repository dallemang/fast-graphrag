#!/usr/bin/env python3
"""
Utility functions for retrying operations that might fail transiently.
"""

import time
import logging
import functools
import random
from typing import Callable, Any, List, Type, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    max_delay: float = 30.0,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable:
    """
    Decorator for retrying functions that may fail with transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for the delay between consecutive retries
        jitter: Whether to add random jitter to the delay
        max_delay: Maximum delay between retries in seconds
        exceptions: List of exception types that should trigger a retry
                  If None, retries on any Exception
    
    Returns:
        Decorator function
    """
    if exceptions is None:
        exceptions = [Exception]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(f"All {max_retries} retry attempts failed for {func.__name__}: {e}")
                        raise last_exception
                    
                    # Calculate delay with optional jitter
                    current_delay = min(delay * (backoff_factor ** attempt), max_delay)
                    if jitter:
                        current_delay = current_delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt+1}/{max_retries} for {func.__name__} "
                        f"failed with error: {e}. Retrying in {current_delay:.2f}s"
                    )
                    time.sleep(current_delay)
            
            # This should never be reached due to the raise in the loop
            raise RuntimeError("Unexpected exit from retry loop")
        
        return wrapper
    
    return decorator

def retry_on_exception(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    max_delay: float = 30.0,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Union[Callable, Any]:
    """
    Alternative form of with_retry that can be used as a decorator with or without arguments.
    
    Usage:
        @retry_on_exception
        def function_with_defaults():
            ...
        
        @retry_on_exception(max_retries=5, exceptions=[ConnectionError])
        def function_with_custom_retries():
            ...
    """
    actual_decorator = with_retry(
        max_retries=max_retries,
        retry_delay=retry_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        max_delay=max_delay,
        exceptions=exceptions
    )
    
    # Called directly with a function
    if func is not None:
        return actual_decorator(func)
    
    # Called with arguments
    return actual_decorator