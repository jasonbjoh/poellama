import os
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import fastapi_poe as fp
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RateLimiter:
    """Simple rate limiter to prevent API overuse."""
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute  # seconds between calls
        self.last_call = 0
    
    async def wait(self):
        """Wait if necessary before making next API call."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            delay = self.interval - elapsed
            logging.info(f"Rate limit: waiting {delay:.1f} seconds before next call")
            await asyncio.sleep(delay)
        self.last_call = time.time()

class ModelTester:
    def __init__(self, calls_per_minute: int = 5):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("POE_API_KEY")
        if not self.api_key:
            raise ValueError("POE_API_KEY environment variable is not set")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(calls_per_minute)
        
        # Load models configuration
        self.models_file = Path(__file__).parent.parent / "models.json"
        logging.info(f"Looking for models.json at: {self.models_file}")
        if not self.models_file.exists():
            raise FileNotFoundError(f"Could not find models.json at {self.models_file}")
            
        with open(self.models_file, 'r') as f:
            self.models_config = json.load(f)
            logging.info(f"Successfully loaded {len(self.models_config['models'])} models from configuration")
            
        self.test_message = "Say 'hello' if you can hear me."
    
    async def test_model(self, model_name: str) -> Dict:
        """Test a single model and return the result."""
        model_config = next(
            (model for model in self.models_config["models"] if model["name"] == model_name),
            None
        )
        
        if not model_config:
            return {
                "model": model_name,
                "success": False,
                "error": f"Model '{model_name}' not found in configuration"
            }
            
        if not model_config["available"]:
            return {
                "model": model_name,
                "success": False,
                "error": f"Model '{model_name}' is marked as unavailable: {model_config.get('description', '')}"
            }
        
        try:
            # Wait for rate limit
            await self.rate_limiter.wait()
            
            logging.info(f"Testing model: {model_name}")
            messages = [fp.ProtocolMessage(role="user", content=self.test_message)]
            full_response = ""
            
            async for partial in fp.get_bot_response(
                messages=messages,
                bot_name=model_config["bot_name"],
                api_key=self.api_key
            ):
                full_response += partial.text
            
            return {
                "model": model_name,
                "success": True,
                "response": full_response.strip()
            }
            
        except Exception as e:
            logging.error(f"Error testing model {model_name}: {str(e)}")
            return {
                "model": model_name,
                "success": False,
                "error": str(e)
            }
    
    async def test_all_models(self, max_concurrent: int = 1) -> List[Dict]:
        """Test all available models and return results."""
        results = []
        available_models = [model for model in self.models_config["models"] if model["available"]]
        
        logging.info(f"Testing {len(available_models)} available models (rate limit: {self.rate_limiter.calls_per_minute} calls/minute)")
        
        for model in available_models:
            result = await self.test_model(model["name"])
            results.append(result)
            
        return results

def print_results(results: List[Dict]):
    """Print test results in a formatted way."""
    print("\nModel Test Results:")
    print("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    
    for result in results:
        model_name = result["model"]
        if result["success"]:
            print(f"✅ {model_name}:")
            print(f"   Response: {result['response']}\n")
            successful_tests += 1
        else:
            print(f"❌ {model_name}:")
            print(f"   Error: {result['error']}\n")
            failed_tests += 1
    
    print("=" * 80)
    print(f"Summary: {successful_tests} successful, {failed_tests} failed")
    print(f"Total models tested: {len(results)}")

async def main():
    try:
        # Initialize tester with rate limit of 5 calls per minute
        tester = ModelTester(calls_per_minute=5)
        
        # Test a specific model
        print("\nTesting single model (Claude-3.5-Sonnet)...")
        result = await tester.test_model("Claude-3.5-Sonnet")
        print_results([result])
        
        # Test all available models
        print("\nTesting all available models...")
        results = await tester.test_all_models()
        print_results(results)
        
    except Exception as e:
        logging.error(f"Test execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 