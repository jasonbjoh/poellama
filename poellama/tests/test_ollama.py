import aiohttp
import asyncio
import json
import time

async def test_tags():
    print("\nTesting /api/tags endpoint...")
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/api/tags') as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print("\nAvailable models and tags:")
                for model in data['models']:
                    print(f"- {model['name']}: {', '.join(model['tags'])}")
            else:
                print("Error:", await response.text())

async def test_generate(stream: bool = False):
    print(f"\nTesting Ollama-compatible /api/generate endpoint (streaming={stream})...")
    test_requests = [
        {
            "model": "Claude-3.5-Haiku",
            "prompt": "What is the capital of France?",
            "system": "You are a helpful assistant.",
            "stream": stream
        },
        {
            "model": "GPT-4o-Mini",
            "prompt": "Write a haiku about programming.",
            "system": "You are a creative writing assistant.",
            "stream": stream
        },
        {
            "model": "Gemini-2.0-Flash-Lite",
            "prompt": "Write a Python function to calculate fibonacci numbers.",
            "system": "You are an expert Python programmer.",
            "stream": stream
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for req in test_requests:
            print(f"\nTesting with model: {req['model']}")
            print(f"Prompt: {req['prompt']}")
            print(f"System: {req['system']}")
            
            async with session.post(
                'http://localhost:8000/api/generate',
                json=req
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    if stream:
                        print("\nStreaming response:")
                        buffer = ""
                        async for line in response.content:
                            chunk = line.decode().strip()
                            if not chunk:
                                continue
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]  # Remove "data: " prefix
                            if chunk == "[DONE]":
                                print("\nStream completed")
                                break
                            try:
                                data = json.loads(chunk)
                                if "choices" in data:  # OpenAI format
                                    content = data["choices"][0]["delta"].get("content", "")
                                    if content:
                                        print(content, end="", flush=True)
                                else:  # Ollama format
                                    response_text = data.get("response", "")
                                    if response_text:
                                        print(response_text, end="", flush=True)
                                    if data.get("done", False):
                                        print("\nStream completed")
                            except json.JSONDecodeError:
                                continue
                    else:
                        data = await response.json()
                        print("\nResponse:")
                        print(f"Model: {data['model']}")
                        print(f"Response text: {data['response'][:200]}...")  # First 200 chars
                        print(f"Done: {data['done']}")
                        print(f"Total duration: {data['total_duration']/1e9:.2f} seconds")  # Convert ns to seconds
                else:
                    print("Error:", await response.text())
            
            # Wait a bit between requests to respect rate limiting
            await asyncio.sleep(2)

async def main():
    try:
        print("Starting Ollama Compatibility Tests")
        print("=================================")
        
        # Test tags endpoint
        await test_tags()
        
        # Test generate endpoint (non-streaming)
        await test_generate(stream=False)
        
        # Test generate endpoint (streaming)
        await test_generate(stream=True)
        
    except aiohttp.ClientError as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 