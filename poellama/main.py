import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
import fastapi_poe as fp
from typing import Dict, Optional, List, Union, Any, Tuple
from pydantic import BaseModel, Field
import time
from datetime import datetime, timezone
from json import JSONDecodeError  # Import JSONDecodeError from json module
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.background import BackgroundTasks
from pathlib import Path
import uuid
import argparse
import uvicorn

# Load models configuration
MODELS_FILE = Path(__file__).parent / "models.json"
with open(MODELS_FILE, 'r') as f:
    MODELS_CONFIG = json.load(f)

# Model mapping from OpenAI model names to Poe model names
MODEL_LOOKUP = {}

# Populate MODEL_LOOKUP from models.json
for model in MODELS_CONFIG["models"]:
    model_name = model["name"]
    bot_name = model["bot_name"]
    # Add the exact name as in models.json
    MODEL_LOOKUP[model_name.lower()] = bot_name
    # Also add without dashes for compatibility
    MODEL_LOOKUP[model_name.lower().replace("-", "")] = bot_name
    # Add the bot_name directly as well
    MODEL_LOOKUP[bot_name.lower()] = bot_name

# Add some common aliases and compatibility mappings
additional_mappings = {
    # Claude models
    "claude-3-opus-20240229": "Claude-3.5-Sonnet",
    "claude-3-sonnet-20240229": "Claude-3.5-Sonnet",
    "claude-3-haiku-20240307": "Claude-3.5-Sonnet",
    "claude-3.5-sonnet-20240620": "Claude-3.5-Sonnet",
    "claude-3.5-haiku-20240307": "Claude-3.5-Sonnet",
    
    # GPT models
    "gpt-3.5-turbo": "GPT-4o",
    "gpt-4": "GPT-4o",
    "gpt-4-turbo": "GPT-4o",
    
    # Shorthand names
    "claude": "Claude-3.5-Sonnet",
    "gpt": "GPT-4o"
}

# Add additional mappings
MODEL_LOOKUP.update(additional_mappings)

def validate_model(model_name: str) -> str:
    """Validate model name and return Poe bot_name if valid."""
    # For Ollama compatibility, we need to handle model names case-insensitively
    model_lower = model_name.lower()
    
    # First check if this is a direct match in our lookup
    if model_name in MODEL_LOOKUP:
        return MODEL_LOOKUP[model_name]
    
    # Then check case-insensitive match
    for key, value in MODEL_LOOKUP.items():
        if key.lower() == model_lower:
            return value
    
    # If we get here, the model wasn't found
    available_models = ", ".join(sorted(set(MODEL_LOOKUP.values())))
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' not found. Available models: {available_models}"
    )

def get_fallback_model(original_model: str) -> str:
    """Get a fallback model if the original model fails."""
    # Define fallback chain - each model has a fallback
    fallback_map = {
        "Claude-3.7-Sonnet": "Claude-3.5-Sonnet",
        "Claude-3.7-Sonnet-Reasoning": "Claude-3.5-Sonnet",
        "Claude-3.5-Sonnet": "GPT-4o",
        "GPT-4o": "GPT-4o-Mini",
        "Gemini-2.0-Pro": "Gemini-2.0-Flash"
    }
    
    # Return the fallback model or a default if no fallback defined
    return fallback_map.get(original_model, Config.DEFAULT_MODEL)

# Model classes for Ollama API compatibility
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict] = None

class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

# OpenAI API compatibility classes
class ContentPart(BaseModel):
    type: str
    text: str

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[ContentPart, Dict[str, str], str]]]
    name: Optional[str] = None

    def get_content_text(self) -> str:
        """Extract text content from various formats"""
        if isinstance(self.content, str):
            return self.content
        
        # Handle list of content parts
        text_parts = []
        for part in self.content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        
        return "".join(text_parts)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# Debug level configuration
class DebugLevel:
    MINIMAL = 0    # Basic info, warnings, and errors
    VERBOSE = 1    # All debug statements

# Set up logging with debug level control
def setup_logging(debug_level: int = DebugLevel.MINIMAL):
    """
    Set up logging with debug level control.
    Creates a Logs directory and uses a unique log filename with date and time.
    
    Args:
        debug_level: DebugLevel.MINIMAL for basic info, DebugLevel.VERBOSE for all debug statements
    """
    import os
    from datetime import datetime
    
    # Create Logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique log filename with date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(logs_dir, f"poe_server_{timestamp}.log")
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    
    # Set log levels based on debug_level
    if debug_level == DebugLevel.VERBOSE:
        file_handler.setLevel(logging.DEBUG)
        root_level = logging.DEBUG
        console_level = logging.DEBUG
        logging.info(f"Logging in VERBOSE mode to {log_filename}")
    else:  # MINIMAL
        file_handler.setLevel(logging.INFO)
        root_level = logging.INFO
        console_level = logging.INFO
        logging.info(f"Logging in MINIMAL mode to {log_filename}")
    
    # Create stream handler with error handling for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=root_level,
        handlers=[file_handler, stream_handler],
        force=True  # Ensure logging is reset
    )
    
    # Set root logger level
    logging.getLogger().setLevel(root_level)
    
    # Add a filter to the stream handler to handle encoding errors
    class EncodingFilter(logging.Filter):
        def filter(self, record):
            try:
                # Filter out GeneratorExit messages from httpcore
                if "receive_response_body.failed exception=GeneratorExit()" in str(record.msg):
                    return False
                
                # Pre-format the message to catch encoding errors
                if isinstance(record.msg, str):
                    # Preserve special characters that might be part of tags
                    if '<' in record.msg or '>' in record.msg:
                        # Use UTF-8 for messages with tags to preserve them
                        record.msg = record.msg.encode('utf-8', errors='replace').decode('utf-8')
                    else:
                        # Use cp1252 for other messages for console compatibility
                        record.msg = record.msg.encode('cp1252', errors='replace').decode('cp1252')
            except Exception as e:
                record.msg = f"[Encoding error in log message: {str(e)}]"
            return True
    
    stream_handler.addFilter(EncodingFilter())
    
    # Log server start with configuration details
    logging.info(f"Starting Poe Local Server with logging level: {'VERBOSE' if debug_level == DebugLevel.VERBOSE else 'MINIMAL'}")
    return log_filename

class Config:
    """Configuration settings for the server"""
    # Rate limiting
    CALLS_PER_MINUTE = int(os.getenv("POE_CALLS_PER_MINUTE", "10"))
    MIN_DELAY_BETWEEN_REQUESTS = 1.0  # Minimum seconds between requests
    
    # Retry settings
    MAX_RETRIES = 3
    ADDITIONAL_DELAY_AFTER_ERROR = 5.0  # Additional seconds to wait after an error
    
    # Testing
    TEST_MODEL = "GPT-4o-Mini"  # Model used for testing (exact bot_name from models.json)
    DEFAULT_MODEL = "GPT-4o-Mini"  # Default model to use if none specified (exact bot_name from models.json)
    INITIAL_BACKOFF = 5.0  # Initial backoff time in seconds (increased)
    BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier

class RateLimiter:
    """Rate limiter for API calls with support for multiple keys"""
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests in seconds
        self.key_data = {}  # Dictionary to store key-specific data
        self.lock = asyncio.Lock()
    
    def register_key(self, key_id: str):
        """Register a new API key with the rate limiter"""
        if key_id not in self.key_data:
            self.key_data[key_id] = {
                "last_request_time": 0,
                "consecutive_errors": 0,
                "backoff_until": 0,
                "total_requests": 0,
                "total_errors": 0,
                "is_priority": False
            }
    
    async def wait(self, key_id: str = None):
        """
        Wait until the next request can be made.
        If key_id is provided, wait for that specific key.
        Otherwise, wait for the next available key.
        """
        if key_id:
            wait_time = await self.wait_for_key(key_id)
            if wait_time > 0:
                logging.info(f"Rate limit: Waiting {wait_time:.2f}s for {key_id} key")
                await asyncio.sleep(wait_time)
            return
        
        # If no specific key is requested, find the key with the shortest wait time
        keys = list(self.key_data.keys())
        if not keys:
            logging.warning("No API keys registered with rate limiter")
            return
        
        best_key, wait_time = self.get_next_available_key(keys)
        if wait_time > 0:
            logging.info(f"Rate limit: Waiting {wait_time:.2f}s for {best_key} key (best available)")
            await asyncio.sleep(wait_time)
    
    async def wait_for_key(self, key_id: str) -> float:
        """
        Wait until the specified key can be used.
        Returns the actual wait time.
        """
        wait_time = await self.get_wait_time(key_id)
        if wait_time > 0:
            logging.debug(f"Rate limit: Waiting {wait_time:.2f}s for {key_id} key")
            await asyncio.sleep(wait_time)
        return wait_time
    
    async def get_wait_time(self, key_id: str) -> float:
        """Calculate how long to wait before using this key"""
        if key_id not in self.key_data:
            logging.warning(f"Key {key_id} not registered with rate limiter")
            self.register_key(key_id)
            return 0
        
        key_info = self.key_data[key_id]
        now = time.time()
        
        # Calculate time since last request
        time_since_last_request = now - key_info['last_request_time']
        
        # Calculate minimum interval between requests
        min_interval = 60.0 / self.requests_per_minute
        
        # Add additional delay if there have been errors
        additional_delay = 0
        if key_info['consecutive_errors'] > 0:
            additional_delay = Config.ADDITIONAL_DELAY_AFTER_ERROR * key_info['consecutive_errors']
            logging.debug(f"Adding {additional_delay:.2f}s delay due to {key_info['consecutive_errors']} consecutive errors for {key_id} key")
        
        # Calculate wait time
        wait_time = max(0, min_interval - time_since_last_request + additional_delay)
        
        # Add backoff time if applicable
        if key_info['backoff_until'] > now:
            backoff_wait = key_info['backoff_until'] - now
            logging.debug(f"Adding {backoff_wait:.2f}s backoff time for {key_id} key")
            wait_time = max(wait_time, backoff_wait)
        
        return wait_time
    
    def get_next_available_key(self, key_ids: List[str]) -> Tuple[str, float]:
        """
        Find the key with the shortest wait time.
        Returns the key ID and the wait time.
        """
        if not key_ids:
            raise ValueError("No keys provided")
        
        best_key = key_ids[0]
        best_wait_time = float('inf')
        
        for key_id in key_ids:
            if key_id not in self.key_data:
                logging.warning(f"Key {key_id} not registered with rate limiter")
                self.register_key(key_id)
                return key_id, 0
            
            key_info = self.key_data[key_id]
            now = time.time()
            
            # Calculate time since last request
            time_since_last_request = now - key_info['last_request_time']
            
            # Calculate minimum interval between requests
            min_interval = 60.0 / self.requests_per_minute
            
            # Add additional delay if there have been errors
            additional_delay = 0
            if key_info['consecutive_errors'] > 0:
                additional_delay = Config.ADDITIONAL_DELAY_AFTER_ERROR * key_info['consecutive_errors']
            
            # Calculate wait time
            wait_time = max(0, min_interval - time_since_last_request + additional_delay)
            
            # Add backoff time if applicable
            if key_info['backoff_until'] > now:
                backoff_wait = key_info['backoff_until'] - now
                wait_time = max(wait_time, backoff_wait)
            
            # Prioritize keys that are marked as priority
            if not key_info['is_priority']:
                wait_time += 0.1  # Small penalty for non-priority keys
            
            # Update best key if this one has a shorter wait time
            if wait_time < best_wait_time:
                best_key = key_id
                best_wait_time = wait_time
        
        if best_wait_time < float('inf'):
            logging.debug(f"Selected {best_key} key with {best_wait_time:.2f}s wait time")
            return best_key, best_wait_time
        
        # If all keys have infinite wait time, return the first one
        logging.warning("All keys have infinite wait time, using first key")
        return key_ids[0], float('inf')
    
    def prioritize_key(self, key_id: str):
        """Mark a key as priority, so it's preferred when selecting keys"""
        if key_id not in self.key_data:
            logging.warning(f"Key {key_id} not registered with rate limiter")
            self.register_key(key_id)
        
        self.key_data[key_id]['is_priority'] = True
        logging.info(f"Prioritized {key_id} key for future requests")
    
    def record_success(self, key_id: str):
        """Record a successful API call with this key"""
        if key_id not in self.key_data:
            logging.warning(f"Key {key_id} not registered with rate limiter")
            self.register_key(key_id)
        
        self.key_data[key_id]['last_request_time'] = time.time()
        self.key_data[key_id]['consecutive_errors'] = 0
        self.key_data[key_id]['total_requests'] += 1
        
        # Log success at debug level
        logging.debug(f"Recorded successful request for {key_id} key (total: {self.key_data[key_id]['total_requests']})")
    
    def record_error(self, key_id: str):
        """
        Record an error with this key.
        This will add a delay to future requests with this key.
        """
        if key_id not in self.key_data:
            logging.warning(f"Key {key_id} not registered with rate limiter")
            self.register_key(key_id)
        
        self.key_data[key_id]['last_request_time'] = time.time()
        self.key_data[key_id]['consecutive_errors'] += 1
        self.key_data[key_id]['total_errors'] += 1
        self.key_data[key_id]['total_requests'] += 1
        
        # Apply exponential backoff
        backoff_time = Config.INITIAL_BACKOFF * (Config.BACKOFF_FACTOR ** (self.key_data[key_id]['consecutive_errors'] - 1))
        self.key_data[key_id]['backoff_until'] = time.time() + backoff_time
        
        # Log error with backoff information
        logging.warning(f"Recorded error for {key_id} key (consecutive: {self.key_data[key_id]['consecutive_errors']}, " +
                      f"total: {self.key_data[key_id]['total_errors']}, " +
                      f"backoff: {backoff_time:.2f}s)")
    
    def get_key_info(self, key_id: str) -> dict:
        """Get information about a key's usage and rate limiting"""
        if key_id not in self.key_data:
            logging.warning(f"Key {key_id} not registered with rate limiter")
            self.register_key(key_id)
        
        key_info = self.key_data[key_id].copy()
        now = time.time()
        
        # Add some derived information
        key_info['time_since_last_request'] = now - key_info['last_request_time']
        key_info['backoff_remaining'] = max(0, key_info['backoff_until'] - now)
        
        # Calculate wait time
        min_interval = 60.0 / self.requests_per_minute
        additional_delay = 0
        if key_info['consecutive_errors'] > 0:
            additional_delay = Config.ADDITIONAL_DELAY_AFTER_ERROR * key_info['consecutive_errors']
        
        wait_time = max(0, min_interval - key_info['time_since_last_request'] + additional_delay)
        if key_info['backoff_until'] > now:
            wait_time = max(wait_time, key_info['backoff_until'] - now)
        
        key_info['wait_time'] = wait_time
        
        return key_info

class PoeClient:
    """Handles all interactions with the Poe API with load balancing between keys"""
    def __init__(self, primary_api_key: str, fallback_api_key: Optional[str], rate_limiter: RateLimiter):
        self.primary_api_key = primary_api_key
        self.fallback_api_key = fallback_api_key
        self.rate_limiter = rate_limiter
        self.using_fallback_key = False  # For backward compatibility
        self.primary_failures = 0
        self.fallback_failures = 0
        self.max_failures_before_switch = 3
        self.key_switch_cooldown = 300
        
        # Register both keys with the rate limiter
        self.rate_limiter.register_key("primary")
        if self.fallback_api_key:
            self.rate_limiter.register_key("fallback")
    
    async def get_best_api_key(self) -> Tuple[str, str]:
        """
        Get the best API key to use based on rate limiting and availability.
        Returns a tuple of (key_id, api_key)
        """
        # If we only have one key, use it
        if not self.fallback_api_key:
            logging.debug("Only primary API key available")
            return "primary", self.primary_api_key
        
        # Get wait times for both keys
        primary_wait = await self.rate_limiter.get_wait_time("primary")
        fallback_wait = await self.rate_limiter.get_wait_time("fallback")
        
        # Get detailed key info for logging
        primary_info = self.rate_limiter.get_key_info("primary")
        fallback_info = self.rate_limiter.get_key_info("fallback")
        
        # Log wait times and key status
        logging.debug(f"API key wait times - Primary: {primary_wait:.2f}s, Fallback: {fallback_wait:.2f}s")
        logging.debug(f"Primary key status: {primary_info['consecutive_errors']} consecutive errors, {primary_info['total_errors']} total errors")
        logging.debug(f"Fallback key status: {fallback_info['consecutive_errors']} consecutive errors, {fallback_info['total_errors']} total errors")
        
        # Check if either key is prioritized
        if primary_info['is_priority']:
            logging.debug("Primary key is prioritized")
            self.using_fallback_key = False
            return "primary", self.primary_api_key
        
        if fallback_info['is_priority']:
            logging.debug("Fallback key is prioritized")
            self.using_fallback_key = True
            return "fallback", self.fallback_api_key
        
        # Determine which key to use based on wait times
        if primary_wait <= fallback_wait:
            # Primary key is available sooner
            if self.using_fallback_key and primary_wait < 1.0:
                logging.info(f"Switching back to primary API key (wait time: {primary_wait:.2f}s vs fallback: {fallback_wait:.2f}s)")
                self.using_fallback_key = False
            return "primary", self.primary_api_key
        else:
            # Fallback key is available sooner
            if not self.using_fallback_key:
                logging.info(f"Switching to fallback API key (wait time: {fallback_wait:.2f}s vs primary: {primary_wait:.2f}s)")
                self.using_fallback_key = True
            return "fallback", self.fallback_api_key

    async def get_response(self, 
                          messages: List[fp.ProtocolMessage], 
                          model: str, 
                          stream_handler = None,
                          max_retries: int = Config.MAX_RETRIES,
                          force_key: Optional[str] = None) -> str:
        """
        Get a response from Poe API with load balancing and retry logic.
        If stream_handler is provided, it will be called for each chunk.
        If force_key is provided, it will use that specific key ("primary" or "fallback").
        """
        full_response = ""
        start_time = time.time()
        is_first = True
        retry_count = 0
        last_error = None
        
        # Log the raw messages being sent to the API (only in verbose mode)
        logging.debug(f"Sending to {model} - Raw messages: {[{'role': m.role, 'content': m.content[:100] + '...' if len(m.content) > 100 else m.content} for m in messages]}")
        
        # If a specific key is forced, use it
        if force_key:
            if force_key == "primary":
                key_id, current_api_key = "primary", self.primary_api_key
            elif force_key == "fallback" and self.fallback_api_key:
                key_id, current_api_key = "fallback", self.fallback_api_key
            else:
                raise ValueError(f"Invalid force_key: {force_key}")
            
            # Log forced key usage
            logging.info(f"Using forced API key: {key_id}")
        else:
            # Otherwise, get the best key based on rate limiting
            key_id, current_api_key = await self.get_best_api_key()
            logging.info(f"Selected API key for request: {key_id}")
        
        # Wait for the selected key to be available
        wait_time = await self.rate_limiter.wait_for_key(key_id)
        if wait_time > 0:
            logging.info(f"Waited {wait_time:.2f}s for {key_id} API key to be available")
        
        # Try to get a response, with retries and key switching if needed
        while retry_count <= max_retries:
            try:
                # Log the attempt
                if retry_count > 0:
                    logging.info(f"Retry {retry_count}/{max_retries} with {key_id} API key")
                
                # Get the bot response generator
                bot_response_generator = fp.get_bot_response(
                    messages=messages,
                    bot_name=model,
                    api_key=current_api_key
                )
                
                # Process each chunk from the generator
                async for partial in bot_response_generator:
                    full_response += partial.text
                    
                    # Sanitize the chunk to handle tag issues
                    sanitized_text = sanitize_streaming_response(partial.text, is_first, False)
                    
                    # Log the chunk if it's the first one or contains potentially interesting content (only in verbose mode)
                    if is_first or any(tag in sanitized_text for tag in ['<', '>', '</plan_mode_response>']):
                        logging.debug(f"Chunk from {model}: {sanitized_text}")
                    
                    if stream_handler:
                        # For the first chunk, we want to make sure it's not empty
                        # to prevent clients from cutting off the beginning of responses
                        if is_first and not sanitized_text.strip():
                            sanitized_text = " "  # Add a space to ensure it's not empty
                            
                        await stream_handler(
                            text=sanitized_text,
                            is_first=is_first,
                            is_last=False,
                            start_time=start_time,
                            model=model
                        )
                        is_first = False
                
                # Record successful API call
                self.rate_limiter.record_success(key_id)
                
                # Reset failure counters if successful
                if key_id == "primary":
                    self.primary_failures = 0
                else:
                    self.fallback_failures = 0
                
                # Log success
                end_time = time.time()
                response_time = end_time - start_time
                logging.info(f"Request to {model} completed successfully using {key_id} API key in {response_time:.2f}s")
                
                # Sanitize the full response for the final return
                sanitized_full_response = sanitize_streaming_response(full_response, False, True)
                return sanitized_full_response
                
            except Exception as e:
                # Record the error
                self.rate_limiter.record_error(key_id)
                last_error = e
                
                # Increment failure counter for the current key
                if key_id == "primary":
                    self.primary_failures += 1
                    logging.warning(f"Primary API key failure ({self.primary_failures}): {str(e)}")
                else:
                    self.fallback_failures += 1
                    logging.warning(f"Fallback API key failure ({self.fallback_failures}): {str(e)}")
                
                # Try to switch keys if available and not already forced
                if not force_key and self.fallback_api_key:
                    if key_id == "primary" and self.primary_failures >= self.max_failures_before_switch:
                        logging.info(f"Switching to fallback API key after {self.primary_failures} primary key failures")
                        key_id, current_api_key = "fallback", self.fallback_api_key
                    elif key_id == "fallback" and self.fallback_failures >= self.max_failures_before_switch:
                        logging.info(f"Switching to primary API key after {self.fallback_failures} fallback key failures")
                        key_id, current_api_key = "primary", self.primary_api_key
                
                # Apply backoff before retry
                retry_count += 1
                if retry_count <= max_retries:
                    backoff_time = Config.INITIAL_BACKOFF * (Config.BACKOFF_FACTOR ** (retry_count - 1))
                    logging.info(f"Backing off for {backoff_time:.2f}s before retry {retry_count}/{max_retries}")
                    await asyncio.sleep(backoff_time)
        
        # If we get here, all retries failed
        error_msg = f"Failed to get response after {max_retries} retries. Last error: {str(last_error)}"
        logging.error(error_msg)
        raise Exception(error_msg)

    async def switch_to_primary_key(self):
        """Switch back to the primary API key after a cooldown period."""
        if self.using_fallback_key:
            logging.info(f"Switching back to primary API key after cooldown")
            self.using_fallback_key = False
            self.primary_failures = 0
            return True
        return False

    async def switch_to_fallback_key(self):
        """Switch to the fallback API key if available."""
        if not self.using_fallback_key and self.fallback_api_key:
            logging.info(f"Switching to fallback API key")
            self.using_fallback_key = True
            self.fallback_failures = 0
            return True
        return False

    def create_messages(self, prompt: str = None, system: str = None, message_list: List[dict] = None) -> List[fp.ProtocolMessage]:
        """
        Convert messages to Poe protocol messages.
        Can accept either a prompt+system or a list of messages in OpenAI format.
        """
        protocol_messages = []
        
        if message_list:
            # Convert OpenAI-style messages to Poe protocol messages
            for msg in message_list:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Handle different content formats
                if isinstance(content, list):
                    # Handle OpenAI's content array format (e.g., with images or structured content)
                    # Extract text parts and join them
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "".join(text_parts)
                
                if not role or not content:
                    continue
                    
                # Map OpenAI roles to Poe roles
                poe_role = role
                if role == "system":
                    poe_role = "system"
                elif role == "user":
                    poe_role = "user"
                elif role == "assistant":
                    poe_role = "bot"
                
                protocol_messages.append(fp.ProtocolMessage(role=poe_role, content=content))
        else:
            # Create from prompt and optional system message
            if system:
                protocol_messages.append(fp.ProtocolMessage(role="system", content=system))
            if prompt:
                protocol_messages.append(fp.ProtocolMessage(role="user", content=prompt))
        
        return protocol_messages
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string"""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)

# Load environment variables and setup
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Poe Local Server",
    description="A local server for Poe API with OpenAI-compatible endpoints",
    version="1.0.0"
)

# Initialize variables that will be set in run_server
poe_client = None
rate_limiter = None

@app.on_event("startup")
async def startup_event():
    """Run startup checks and tests"""
    global poe_client
    
    if not poe_client:
        # This should not happen in normal operation, but just in case
        logging.warning("PoeClient not initialized before startup event. Initializing now.")
        initialize_api_clients()
    
    logging.info("Running startup checks")
    
    # Test API keys
    await test_api_keys()
    
    # Start periodic key check task
    asyncio.create_task(periodic_key_check())

def initialize_api_clients():
    """Initialize the API clients with the appropriate keys"""
    global poe_client, rate_limiter, FALLBACK_POE_API_KEY
    
    # Poe API keys
    POE_API_KEY = os.getenv("POE_API_KEY")
    if not POE_API_KEY:
        raise ValueError("POE_API_KEY environment variable is not set")

    # Fallback API key (optional)
    FALLBACK_POE_API_KEY = os.getenv("FALLBACK_POE_API_KEY")
    if FALLBACK_POE_API_KEY:
        logging.info("Fallback Poe API key is configured")
    else:
        logging.info("No fallback Poe API key configured")

    # Initialize rate limiter and Poe client
    rate_limiter = RateLimiter(requests_per_minute=Config.CALLS_PER_MINUTE)
    poe_client = PoeClient(
        primary_api_key=POE_API_KEY,
        fallback_api_key=FALLBACK_POE_API_KEY,
        rate_limiter=rate_limiter
    )

# Ollama-compatible endpoints
@app.post("/api/generate")
async def generate(request: OllamaGenerateRequest) -> OllamaResponse:
    try:
        logging.info(f"Processing Ollama-compatible generate request for model: {request.model}")
        
        # Validate model and get bot name
        try:
            model_name = validate_model(request.model)
            logging.debug(f"Mapped '{request.model}' to Poe model '{model_name}'")
        except HTTPException as e:
            # For Ollama compatibility, try with different casing
            try:
                # Try with title case (Claude-3.5-Haiku)
                title_case_model = "-".join(word.capitalize() for word in request.model.replace("-", " ").split())
                model_name = validate_model(title_case_model)
                logging.debug(f"Mapped '{request.model}' to '{title_case_model}' to Poe model '{model_name}'")
            except HTTPException:
                # If that fails, try with the model name as-is
                model_name = request.model
                logging.warning(f"Using model name '{model_name}' directly without validation")
        
        # Create messages
        messages = poe_client.create_messages(
            prompt=request.prompt,
            system=request.system
        )
        
        start_time = time.time_ns()
        load_time = time.time_ns()
        
        # Get response
        full_response = await poe_client.get_response(
            messages=messages,
            model=model_name
        )
        
        end_time = time.time_ns()
        
        # Calculate durations
        total_duration = end_time - start_time
        load_duration = load_time - start_time
        eval_duration = end_time - load_time
        
        # Get current time in ISO format with UTC timezone
        current_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z"
        
        return OllamaResponse(
            model=request.model,
            created_at=current_time,
            response=full_response,
            done=True,
            context=[],
            total_duration=total_duration,
            load_duration=load_duration,
            prompt_eval_count=poe_client.estimate_tokens(request.prompt),
            prompt_eval_duration=0,
            eval_count=poe_client.estimate_tokens(full_response),
            eval_duration=eval_duration
        )
        
    except Exception as e:
        logging.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Original Poe endpoints
@app.get("/models")
async def get_models():
    try:
        logging.info("Fetching available models")
        return {
            "models": [
                {
                    "name": model["name"],
                    "cost": f"${model['dollar_cost']:.5f} per 1k tokens ({model['token_cost']} tokens)",
                    "contextWindow": model["context_window"],
                    "description": model["description"],
                    "available": model["available"]
                }
                for model in MODELS_CONFIG["models"]
            ]
        }
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def sanitize_for_log(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for logging to avoid encoding errors.
    Also truncates long text to prevent log bloat.
    """
    if not text:
        return ""
    
    # Truncate long text
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Replace problematic characters
    try:
        # Test if the string can be encoded in cp1252 (Windows console encoding)
        text.encode('cp1252')
    except UnicodeEncodeError:
        # If not, replace problematic characters
        text = text.encode('cp1252', errors='replace').decode('cp1252')
    
    return text

def sanitize_response(text: str) -> str:
    """
    Sanitize response text to ensure proper tag formatting.
    Handles issues with unmatched tags and encoding.
    """
    if not text:
        return ""
    
    # Ensure consistent UTF-8 encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Check for unmatched closing tags
    if '</plan_mode_response>' in text and '<plan_mode_response>' not in text:
        # Add opening tag at the beginning if there's a closing tag without an opening tag
        text = '<plan_mode_response>' + text
        logging.debug(f"Added missing opening <plan_mode_response> tag to response")
    
    # Check for other common tag issues
    if '</act_mode_response>' in text and '<act_mode_response>' not in text:
        text = '<act_mode_response>' + text
        logging.debug(f"Added missing opening <act_mode_response> tag to response")
    
    # Log any tag-related modifications for debugging
    if '<' in text or '>' in text:
        logging.debug(f"Response contains tags after sanitization: {text[:100]}...")
    
    return text

# Global tag state for streaming responses
class TagState:
    def __init__(self):
        self.buffer = ""  # Buffer for partial tags
        self.open_tags = []  # Stack of open tags
        self.reset()
    
    def reset(self):
        """Reset the tag state for a new response"""
        self.buffer = ""
        self.open_tags = []

# Create a global tag state instance
tag_state = TagState()

def sanitize_streaming_response(text: str, is_first: bool = False, is_last: bool = False) -> str:
    """
    Sanitize streaming response text to ensure proper tag formatting.
    Maintains state across chunks to handle split tags.
    
    Args:
        text: The current chunk of text
        is_first: Whether this is the first chunk of the response
        is_last: Whether this is the last chunk of the response
    
    Returns:
        Sanitized text with proper tag handling
    """
    global tag_state
    
    if not text:
        return ""
    
    # Reset state for new responses
    if is_first:
        tag_state.reset()
    
    # Ensure consistent UTF-8 encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Combine with any buffered text from previous chunks
    combined_text = tag_state.buffer + text
    tag_state.buffer = ""
    
    # Process the combined text to handle tags
    result = ""
    i = 0
    while i < len(combined_text):
        # Look for tag start
        if combined_text[i] == '<':
            # Find the end of this tag
            tag_end = combined_text.find('>', i)
            
            # If we don't find the end of the tag in this chunk
            if tag_end == -1:
                # Buffer the partial tag for the next chunk
                tag_state.buffer = combined_text[i:]
                break
            
            # Extract the full tag
            tag = combined_text[i:tag_end+1]
            
            # Check if it's a closing tag
            if tag.startswith('</'):
                tag_name = tag[2:-1]  # Remove </> to get the tag name
                
                # If we have a closing tag without a matching opening tag
                if tag_name not in tag_state.open_tags:
                    # Add the corresponding opening tag at the beginning of the result
                    opening_tag = f"<{tag_name}>"
                    result = opening_tag + result
                    logging.debug(f"Added missing opening tag: {opening_tag}")
                else:
                    # Remove the tag from our open tags stack
                    while tag_state.open_tags and tag_state.open_tags[-1] != tag_name:
                        tag_state.open_tags.pop()
                    if tag_state.open_tags:
                        tag_state.open_tags.pop()
            
            # Check if it's an opening tag (not self-closing)
            elif not tag.endswith('/>') and not tag.startswith('<!'):
                tag_name = tag[1:-1].split()[0]  # Get the tag name, ignoring attributes
                tag_state.open_tags.append(tag_name)
            
            # Add the tag to the result
            result += tag
            i = tag_end + 1
        else:
            # Regular character, add to result
            result += combined_text[i]
            i += 1
    
    # For the last chunk, close any remaining open tags
    if is_last and tag_state.open_tags:
        for tag in reversed(tag_state.open_tags):
            result += f"</{tag}>"
            logging.debug(f"Added missing closing tag: </{tag}>")
        tag_state.open_tags = []
    
    # Log any tag-related modifications for debugging
    if '<' in result or '>' in result:
        logging.debug(f"Response contains tags after streaming sanitization: {result[:100]}...")
    
    return result

@app.post("/chat")
async def chat(model: str = Config.DEFAULT_MODEL, message: Optional[str] = None):
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        # Validate model
        model_config = validate_model(model)
        model_name = model_config["bot_name"]
        
        logging.info(f"Processing chat request for model: {model_name}")
        logging.debug(f"Message content: {sanitize_for_log(message)}")
        
        messages = poe_client.create_messages(prompt=message)
        try:
            full_response = await poe_client.get_response(
                messages=messages,
                model=model_name
            )
        except fp.client.BotError as e:
            error_data = json.loads(str(e))
            allow_retry = error_data.get("allow_retry", False)
            
            # Try fallback model if retry is allowed
            if allow_retry:
                fallback_model = get_fallback_model(model)
                if fallback_model != model:
                    logging.info(f"Attempting fallback to model: {fallback_model}")
                    
                    # Add extra delay before trying fallback model
                    await asyncio.sleep(Config.MIN_DELAY_BETWEEN_REQUESTS)
                    
                    try:
                        fallback_model_config = validate_model(fallback_model)
                        fallback_bot_name = fallback_model_config["bot_name"]
                        
                        full_response = await poe_client.get_response(
                            messages=messages,
                            model=fallback_bot_name
                        )
                        
                        logging.info(f"Successfully processed chat request with fallback model: {fallback_model}")
                        return {
                            "response": full_response,
                            "model": fallback_model,
                            "fallback_used": True,
                            "original_model": model
                        }
                    except Exception as fallback_error:
                        logging.error(f"Fallback model also failed: {str(fallback_error)}")
            
            # If we get here, either fallback wasn't attempted or it failed
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Bot error",
                    "model": model_name,
                    "message": error_data.get("text", str(e)),
                    "allow_retry": error_data.get("allow_retry", False)
                }
            )
        
        logging.info("Successfully processed chat request")
        logging.debug(f"Response: {sanitize_for_log(full_response[:100])}...")
        
        return {"response": full_response}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tags")
async def get_tags():
    """List available model tags/versions"""
    try:
        logging.info("Fetching model tags")
        return {
            "models": [
                {
                    "name": model["name"],
                    "tags": model["tags"]
                }
                for model in MODELS_CONFIG["models"]
            ]
        }
    except Exception as e:
        logging.error(f"Error fetching model tags: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/switch_key")
async def switch_api_key(key_type: str = "primary"):
    """Manually switch to a specific API key"""
    if key_type not in ["primary", "fallback"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid key_type. Must be 'primary' or 'fallback'"}
        )
    
    if key_type == "fallback" and not FALLBACK_POE_API_KEY:
        return JSONResponse(
            status_code=400,
            content={"error": "No fallback API key configured"}
        )
    
    # Get current key
    current_key_id, _ = await poe_client.get_best_api_key()
    
    # If already using the requested key, just return success
    if current_key_id == key_type:
        return {
            "success": True,
            "message": f"Already using {key_type} API key",
            "current_key": key_type
        }
    
    # Force the next request to use the specified key
    poe_client.rate_limiter.prioritize_key(key_type)
    
    return {
        "success": True,
        "message": f"Switched to {key_type} API key",
        "current_key": key_type,
        "previous_key": current_key_id
    }

@app.get("/api/current_key")
async def get_current_key():
    """Get information about the current API key in use"""
    key_id, _ = await poe_client.get_best_api_key()
    
    # Get wait times for both keys
    primary_wait = await poe_client.rate_limiter.get_wait_time("primary")
    fallback_wait = None
    if FALLBACK_POE_API_KEY:
        fallback_wait = await poe_client.rate_limiter.get_wait_time("fallback")
    
    return {
        "current_key": key_id,
        "fallback_key_available": FALLBACK_POE_API_KEY is not None,
        "wait_times": {
            "primary": round(primary_wait, 2),
            "fallback": round(fallback_wait, 2) if fallback_wait is not None else None
        },
        "rate_limits": {
            "primary": poe_client.rate_limiter.get_key_info("primary"),
            "fallback": poe_client.rate_limiter.get_key_info("fallback") if FALLBACK_POE_API_KEY else None
        }
    }

async def test_api_keys():
    """Test both API keys at startup"""
    # Test primary API key
    logging.info("Testing primary API key...")
    try:
        test_message = [fp.ProtocolMessage(role="user", content="Hello, this is a test message. Please respond with 'OK'.")]
        response = await poe_client.get_response(
            messages=test_message,
            model=Config.TEST_MODEL,
            force_key="primary"
        )
        logging.info(f"Primary API key test successful: {sanitize_for_log(response)}")
    except Exception as e:
        logging.error(f"Primary API key test failed: {str(e)}")
        if not FALLBACK_POE_API_KEY:
            logging.critical("No fallback API key available. Server may not function correctly.")
    
    # Test fallback API key if available
    if FALLBACK_POE_API_KEY:
        logging.info("Testing fallback API key...")
        try:
            test_message = [fp.ProtocolMessage(role="user", content="Hello, this is a test message. Please respond with 'OK'.")]
            response = await poe_client.get_response(
                messages=test_message,
                model=Config.TEST_MODEL,
                force_key="fallback"
            )
            logging.info(f"Fallback API key test successful: {sanitize_for_log(response)}")
        except Exception as e:
            logging.error(f"Fallback API key test failed: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = await request.json()
    except JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
        )

    # Extract and validate parameters
    model = data.get("model", "")
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    temperature = data.get("temperature", 1.0)
    max_tokens = data.get("max_tokens")
    
    # Validate model
    if not model:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing model parameter", "type": "invalid_request_error"}},
        )
    
    # Map to Poe model if needed
    try:
        poe_model = validate_model(model)
    except HTTPException:
        # If validation fails, try with the model name directly
        poe_model = model
        logging.warning(f"Using model name '{poe_model}' directly without validation")
    
    # Validate messages
    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing messages parameter", "type": "invalid_request_error"}},
        )
    
    # Preprocess messages to ensure compatibility
    try:
        # Normalize message format for compatibility
        normalized_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Skip empty messages
            if not role:
                continue
                
            # Handle different content formats
            if isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "".join(text_parts)
            
            normalized_messages.append({"role": role, "content": content})
        
        # Convert to Poe protocol messages
        protocol_messages = poe_client.create_messages(message_list=normalized_messages)
    except Exception as e:
        logging.error(f"Error processing messages: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid messages format: {str(e)}", "type": "invalid_request_error"}},
        )
    
    # Handle streaming response
    if stream:
        # Create a streaming response
        async def generate():
            # Initialize response
            response_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            
            # Helper function to create and yield a chunk
            async def yield_chunk(text, is_first, is_last, model):
                delta = {}
                finish_reason = None
                
                if is_first:
                    # First chunk, send role
                    delta = {"role": "assistant"}
                elif is_last:
                    # Last chunk, send finish reason
                    finish_reason = "stop"
                else:
                    # Content chunk
                    delta = {"content": text}
                
                # Calculate usage (approximate)
                prompt_tokens = poe_client.estimate_tokens(str(normalized_messages))
                completion_tokens = poe_client.estimate_tokens(text) if not is_first and not is_last else 0
                
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason
                        }
                    ]
                }
                
                if is_last:
                    # Add usage info to the last chunk
                    chunk["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                
                yield f"data: {json.dumps(chunk)}\n\n"
                
                if is_last:
                    yield "data: [DONE]\n\n"
            
            try:
                # Log the raw incoming messages for streaming
                logging.info(f"Raw incoming messages to streaming chat_completions endpoint: {json.dumps(normalized_messages, indent=2)}")
                logging.info(f"Using model (streaming): {poe_model}")
                
                # Get the bot response generator
                bot_response_generator = fp.get_bot_response(
                    messages=protocol_messages,
                    bot_name=poe_model,
                    api_key=poe_client.primary_api_key if not poe_client.using_fallback_key else poe_client.fallback_api_key
                )
                
                # Process the generator directly without awaiting it
                full_response = ""
                start_time = time.time()
                is_first = True
                
                # Process each chunk from the generator
                async for partial in bot_response_generator:
                    full_response += partial.text
                    
                    # Sanitize the chunk to handle tag issues
                    sanitized_text = sanitize_streaming_response(partial.text, is_first, False)
                    
                    # Log the chunk if it's the first one or contains potentially interesting content
                    if is_first or any(tag in sanitized_text for tag in ['<', '>', '</plan_mode_response>']):
                        logging.debug(f"Chunk from {poe_model}: {sanitized_text}")
                    
                    # Yield the formatted chunk with sanitized text
                    yield f"data: {json.dumps({
                        'id': response_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'role': 'assistant', 'content': sanitized_text} if is_first else {'content': sanitized_text},
                                'finish_reason': None
                            }
                        ]
                    })}\n\n"
                    
                    is_first = False
                
                # Send the final chunk with usage info
                prompt_tokens = poe_client.estimate_tokens(str(normalized_messages))
                completion_tokens = poe_client.estimate_tokens(full_response)
                
                # Sanitize the full response for the final chunk
                sanitized_full_response = sanitize_streaming_response(full_response, False, True)
                
                # Log the complete response from streaming
                logging.info(f"Complete streaming response from {poe_model}:\n{sanitized_full_response}")
                
                yield f"data: {json.dumps({
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'choices': [
                        {
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop'
                        }
                    ],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    }
                })}\n\n"
                
                yield "data: [DONE]\n\n"
                
                # Record successful API call
                key_id = "fallback" if poe_client.using_fallback_key else "primary"
                poe_client.rate_limiter.record_success(key_id)
                
            except Exception as e:
                # Handle errors in streaming
                error_msg = str(e)
                logging.error(f"Error in streaming response: {error_msg}")
                
                error_json = {
                    "error": {
                        "message": f"Error from Poe API: {error_msg}",
                        "type": "api_error"
                    }
                }
                yield f"data: {json.dumps(error_json)}\n\n"
                yield "data: [DONE]\n\n"
        
        # Return the streaming response
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # Handle non-streaming response
    try:
        # Log the raw incoming messages
        logging.info(f"Raw incoming messages to chat_completions endpoint: {json.dumps(normalized_messages, indent=2)}")
        logging.info(f"Using model: {poe_model}")
        
        # Get response from Poe API
        response = await poe_client.get_response(
            messages=protocol_messages,
            model=poe_model
        )
        
        # Sanitize the response to handle tag issues
        sanitized_response = sanitize_streaming_response(response, True, True)
        
        # Log the raw response
        logging.info(f"Raw response from {poe_model}:\n{sanitized_response}")
        
        # Calculate usage (approximate)
        prompt_tokens = poe_client.estimate_tokens(str(normalized_messages))
        completion_tokens = poe_client.estimate_tokens(sanitized_response)
        
        # Format response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": sanitized_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        return response
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error in chat completion: {error_msg}")
        
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Error from Poe API: {error_msg}", "type": "api_error"}},
        )

async def periodic_key_check():
    """Periodically check and potentially switch API keys"""
    global FALLBACK_POE_API_KEY
    
    while True:
        try:
            # Get wait times for both keys
            primary_wait = await poe_client.rate_limiter.get_wait_time("primary")
            fallback_wait = float('inf')
            if FALLBACK_POE_API_KEY:
                fallback_wait = await poe_client.rate_limiter.get_wait_time("fallback")
            
            # Get current key and detailed key info
            current_key_id, _ = await poe_client.get_best_api_key()
            primary_info = poe_client.rate_limiter.get_key_info("primary")
            fallback_info = poe_client.rate_limiter.get_key_info("fallback") if FALLBACK_POE_API_KEY else None
            
            # Log basic info at INFO level (always visible)
            logging.info(f"API Key Status - Using: {current_key_id}, " +
                         f"Primary wait: {primary_wait:.2f}s, " +
                         f"Fallback wait: {fallback_wait:.2f}s")
            
            # Log detailed info at DEBUG level (only in verbose mode)
            logging.debug(f"Primary key details: {primary_info}")
            if fallback_info:
                logging.debug(f"Fallback key details: {fallback_info}")
            
            # Log load balancing metrics
            primary_success_rate = 100.0
            if primary_info['total_requests'] > 0:
                primary_success_rate = 100.0 * (primary_info['total_requests'] - primary_info['total_errors']) / primary_info['total_requests']
            
            logging.info(f"Load balancing metrics - " +
                         f"Primary: {primary_info['total_requests']} requests, " +
                         f"{primary_success_rate:.1f}% success rate")
            
            if fallback_info and fallback_info['total_requests'] > 0:
                fallback_success_rate = 100.0 * (fallback_info['total_requests'] - fallback_info['total_errors']) / fallback_info['total_requests']
                logging.info(f"Fallback: {fallback_info['total_requests']} requests, " +
                             f"{fallback_success_rate:.1f}% success rate")
            
            # Sleep for the cooldown period
            await asyncio.sleep(poe_client.key_switch_cooldown)
        except Exception as e:
            logging.error(f"Error in periodic key check: {str(e)}")
            # Sleep for a minute before trying again
            await asyncio.sleep(60)

async def test_poe_connection():
    """Test the Poe API connection with a simple query."""
    try:
        print("Testing Poe API connection...")
        messages = poe_client.create_messages(prompt="Say 'hello' if you can hear me.")
        
        # Use a more reliable model for testing
        test_model = "GPT-4o-Mini"  # This model tends to be more reliable
        
        # Test primary API key
        try:
            # Ensure we're using the primary key for the test
            if poe_client.using_fallback_key:
                await poe_client.switch_to_primary_key()
                
            print("Testing with primary API key...")
            full_response = await poe_client.get_response(
                messages=messages,
                model=test_model,
                max_retries=1  # Limit retries for startup test
            )
            
            print(f"Primary API key test successful! Response: {full_response}")
            primary_key_works = True
        except Exception as e:
            print(f"Primary API key test failed: {str(e)}")
            primary_key_works = False
            
        # Test fallback API key if available
        fallback_key_works = False
        if FALLBACK_POE_API_KEY:
            try:
                # Force using the fallback key for this test
                await poe_client.switch_to_fallback_key()
                
                print("Testing with fallback API key...")
                full_response = await poe_client.get_response(
                    messages=messages,
                    model=test_model,
                    max_retries=1
                )
                
                print(f"Fallback API key test successful! Response: {full_response}")
                fallback_key_works = True
                
                # Switch back to primary if it works
                if primary_key_works:
                    await poe_client.switch_to_primary_key()
            except Exception as e:
                print(f"Fallback API key test failed: {str(e)}")
                # If primary key works, switch back to it
                if primary_key_works:
                    await poe_client.switch_to_primary_key()
        
        # Try an alternative model if both keys failed with the first model
        if not primary_key_works and not fallback_key_works:
            alternative_model = "Claude-3.5-Sonnet"  # Another generally reliable model
            print(f"Both API keys failed with {test_model}. Trying alternative model {alternative_model}...")
            
            # Try with primary key first
            try:
                await poe_client.switch_to_primary_key()
                full_response = await poe_client.get_response(
                    messages=messages,
                    model=alternative_model,
                    max_retries=1
                )
                print(f"Alternative model test with primary key successful! Response: {full_response}")
                return True
            except Exception as alt_e:
                print(f"Alternative model with primary key also failed: {str(alt_e)}")
                
                # Try with fallback key if available
                if FALLBACK_POE_API_KEY:
                    try:
                        await poe_client.switch_to_fallback_key()
                        full_response = await poe_client.get_response(
                            messages=messages,
                            model=alternative_model,
                            max_retries=1
                        )
                        print(f"Alternative model test with fallback key successful! Response: {full_response}")
                        return True
                    except Exception as alt_fallback_e:
                        print(f"Alternative model with fallback key also failed: {str(alt_fallback_e)}")
                        return False
                return False
        
        return primary_key_works or fallback_key_works
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

def run_server():
    """Entry point for the console script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Poe Local Server')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    args = parser.parse_args()
    
    # Set up logging with the appropriate debug level
    debug_level = DebugLevel.VERBOSE if args.verbose else DebugLevel.MINIMAL
    log_filename = setup_logging(debug_level=debug_level)
    
    # Log server configuration
    logging.info(f"Server starting on {args.host}:{args.port}")
    logging.info(f"Log file: {log_filename}")
    
    # Initialize API clients
    initialize_api_clients()
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    run_server()