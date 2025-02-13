import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import fastapi_poe as fp
from typing import Dict, Optional, List, Union, Any
from pydantic import BaseModel, Field
import time
from datetime import datetime, timezone
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from pathlib import Path

# Load models configuration
MODELS_FILE = Path(__file__).parent / "models.json"
with open(MODELS_FILE, 'r') as f:
    MODELS_CONFIG = json.load(f)

# Create a lookup dictionary for quick model validation
MODEL_LOOKUP = {model["name"]: model for model in MODELS_CONFIG["models"]}

def validate_model(model_name: str) -> Dict:
    """Validate model name and return model config if valid."""
    if model_name not in MODEL_LOOKUP:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {', '.join(MODEL_LOOKUP.keys())}"
        )
    model = MODEL_LOOKUP[model_name]
    if not model["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is currently unavailable: {model.get('description', '')}"
        )
    return model

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
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None

    def get_content_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)

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
def setup_logging(debug_level: int = DebugLevel.VERBOSE):
    handlers = [
        logging.FileHandler('poe_server.log'),
        logging.StreamHandler()
    ]
    
    # Set all handlers to DEBUG level
    for handler in handlers:
        handler.setLevel(logging.DEBUG)
        
    # Main logger configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Ensure logging is reset
    )
    
    # Set root logger to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

class Config:
    """Configuration for POE API behavior."""
    CALLS_PER_MINUTE = 5  # Rate limit for API calls
    DEFAULT_MODEL = "Claude-3.5-Haiku"  # Default model to use if none specified

class RateLimiter:
    """Simple rate limiter to prevent API overuse."""
    def __init__(self, calls_per_minute: int = Config.CALLS_PER_MINUTE):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute  # seconds between calls
        self.last_call = 0
    
    async def wait(self):
        """Wait if necessary before making next API call."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            delay = self.interval - elapsed
            await asyncio.sleep(delay)
        self.last_call = time.time()

async def test_poe_connection():
    """Test the Poe API connection with a simple query."""
    try:
        print("Testing Poe API connection...")
        messages = [fp.ProtocolMessage(role="user", content="Say 'hello' if you can hear me.")]
        full_response = ""
        
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name="Claude-3.5-Haiku",  # Using Claude-3.5-Haiku for testing
            api_key=POE_API_KEY
        ):
            full_response += partial.text
        
        print(f"Test successful! Response: {full_response}")
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

# Load environment variables and setup
load_dotenv()
setup_logging()

# Poe API key
POE_API_KEY = os.getenv("POE_API_KEY")
if not POE_API_KEY:
    raise ValueError("POE_API_KEY environment variable is not set")

app = FastAPI()
rate_limiter = RateLimiter()

# After the model classes but before the endpoints

class PoeClient:
    """Handles all interactions with the Poe API"""
    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter

    async def get_response(self, 
                          messages: List[fp.ProtocolMessage], 
                          model: str, 
                          stream_handler = None) -> str:
        """
        Get a response from Poe API.
        If stream_handler is provided, it will be called for each chunk with (chunk_text, is_first, is_last)
        Otherwise, returns the complete response as a string
        """
        await self.rate_limiter.wait()
        
        full_response = ""
        start_time = time.time()
        is_first = True
        
        try:
            async for partial in fp.get_bot_response(
                messages=messages,
                bot_name=model,
                api_key=self.api_key
            ):
                full_response += partial.text
                
                if stream_handler:
                    await stream_handler(
                        text=partial.text,
                        is_first=is_first,
                        is_last=False,
                        start_time=start_time,
                        model=model
                    )
                    is_first = False
            
            if stream_handler:
                await stream_handler(
                    text="",
                    is_first=False,
                    is_last=True,
                    start_time=start_time,
                    model=model
                )
            
            return full_response
            
        except Exception as e:
            logging.error(f"Error in Poe API call: {str(e)}")
            raise

    @staticmethod
    def create_messages(
        prompt: Optional[str] = None, 
        system: Optional[str] = None,
        message_list: Optional[List[ChatMessage]] = None
    ) -> List[fp.ProtocolMessage]:
        """Create Poe protocol messages from various input formats"""
        messages = []
        
        if system:
            messages.append(fp.ProtocolMessage(role="system", content=system))
        
        if message_list:
            for msg in message_list:
                role = "system" if msg.role == "system" else "user"
                messages.append(fp.ProtocolMessage(
                    role=role,
                    content=msg.get_content_text()
                ))
        elif prompt:
            messages.append(fp.ProtocolMessage(role="user", content=prompt))
            
        if not messages:
            raise ValueError("Either prompt or message_list must be provided")
            
        return messages

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough approximation of token count"""
        return len(text.split())

# Initialize the Poe client
poe_client = PoeClient(POE_API_KEY, rate_limiter)

# Ollama-compatible endpoints
@app.post("/api/generate")
async def generate(request: OllamaGenerateRequest) -> OllamaResponse:
    try:
        logging.info(f"Processing Ollama-compatible generate request for model: {request.model}")
        
        # Validate model and get bot name
        model_config = validate_model(request.model)
        model_name = model_config["bot_name"]
        
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
            model=model_name  # Use bot_name from config
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

@app.post("/chat")
async def chat(model: str = Config.DEFAULT_MODEL, message: Optional[str] = None):
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        # Validate model
        model_config = validate_model(model)
        model_name = model_config["bot_name"]
        
        logging.info(f"Processing chat request for model: {model_name}")
        logging.debug(f"Message content: {message[:100]}...")
        
        messages = poe_client.create_messages(prompt=message)
        try:
            full_response = await poe_client.get_response(
                messages=messages,
                model=model_name
            )
        except fp.client.BotError as e:
            error_data = json.loads(str(e))
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
        logging.debug(f"Response: {full_response[:100]}...")
        
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

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(raw_request: Request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Parse and validate request
        body = await raw_request.body()
        body_str = body.decode()
        logging.debug(f"Raw request body: {body_str}")  # Use debug level for sensitive data
        
        try:
            json_data = json.loads(body_str)
            request = ChatCompletionRequest(**json_data)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        except Exception as e:
            logging.error(f"Request validation error: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        
        # Validate model and get bot name
        model_config = validate_model(request.model)
        model_name = model_config["bot_name"]
        
        # Create messages
        messages = poe_client.create_messages(message_list=request.messages)
        
        # Handle streaming
        if request.stream:
            async def stream_generator():
                start_time = time.time()
                
                # Send initial role message
                chunk = {
                    "id": f"chatcmpl-{int(start_time)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                
                # Stream the response
                try:
                    async for partial in fp.get_bot_response(
                        messages=messages,
                        bot_name=request.model,
                        api_key=POE_API_KEY
                    ):
                        if partial.text:  # Only send non-empty chunks
                            chunk = {
                                "id": f"chatcmpl-{int(start_time)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": partial.text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                    
                    # Send the final chunk
                    chunk = {
                        "id": f"chatcmpl-{int(start_time)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                except Exception as e:
                    logging.error(f"Error in streaming response: {str(e)}")
                    raise
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # Non-streaming response
        full_response = await poe_client.get_response(
            messages=messages,
            model=model_name  # Use bot_name from config
        )
        
        # Calculate token counts
        prompt_tokens = sum(poe_client.estimate_tokens(msg.get_content_text()) for msg in request.messages)
        completion_tokens = poe_client.estimate_tokens(full_response)
        
        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
        return response.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat completions endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Run when the server starts up"""
    success = await test_poe_connection()
    if not success:
        raise RuntimeError("Failed to connect to Poe API")

def run_server():
    """Entry point for the console script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()