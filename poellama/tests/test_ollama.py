import aiohttp
import asyncio
import json
import time
from datetime import datetime
import sys

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

async def test_current_key():
    """Test the current key status endpoint"""
    print("\nTesting /api/current_key endpoint...")
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/api/current_key') as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print("\nCurrent key status:")
                print(f"- Current key: {data['current_key']}")
                print(f"- Fallback key available: {data['fallback_key_available']}")
                print("\nWait times:")
                print(f"- Primary: {data['wait_times']['primary']} seconds")
                if data['wait_times']['fallback'] is not None:
                    print(f"- Fallback: {data['wait_times']['fallback']} seconds")
                
                print("\nRate limits:")
                print("- Primary key:")
                for k, v in data['rate_limits']['primary'].items():
                    print(f"  - {k}: {v}")
                
                if data['rate_limits']['fallback']:
                    print("- Fallback key:")
                    for k, v in data['rate_limits']['fallback'].items():
                        print(f"  - {k}: {v}")
                
                return data
            else:
                print("Error:", await response.text())
                return None

async def switch_key(key_type):
    """Test switching between keys"""
    print(f"\nSwitching to {key_type} key...")
    async with aiohttp.ClientSession() as session:
        async with session.post(f'http://localhost:8000/api/switch_key?key_type={key_type}') as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Result: {data['message']}")
                return data
            else:
                print("Error:", await response.text())
                return None

async def test_generate(model="GPT-4o-Mini", prompt="What is the capital of France?", system="You are a helpful assistant.", stream=False):
    """Test the generate endpoint with specific parameters"""
    print(f"\nTesting generate with model: {model}")
    print(f"Prompt: {prompt}")
    
    request = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": stream
    }
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/api/generate',
            json=request
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
                    print(f"Response text: {data['response'][:100]}...")  # First 100 chars
                    print(f"Done: {data['done']}")
                    print(f"Total duration: {data['total_duration']/1e9:.2f} seconds")  # Convert ns to seconds
                
                end_time = time.time()
                print(f"Total request time: {end_time - start_time:.2f} seconds")
                return end_time - start_time
            else:
                print("Error:", await response.text())
                return None

async def test_chat_completions(stream=False, model="GPT-4o-Mini"):
    """Test the OpenAI-compatible chat completions endpoint"""
    print(f"\nTesting OpenAI-compatible /v1/chat/completions endpoint (streaming={stream}) with model: {model}...")
    
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": stream
    }
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/v1/chat/completions',
            json=request
        ) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                if stream:
                    print("\nStreaming response:")
                    full_response = ""
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
                            if "choices" in data:
                                content = data["choices"][0]["delta"].get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
                    
                    print(f"\nFull response length: {len(full_response)} characters")
                else:
                    data = await response.json()
                    print("\nResponse:")
                    if "choices" in data:
                        content = data["choices"][0]["message"]["content"]
                        print(f"Content: {content[:100]}...")  # First 100 chars
                        print(f"Usage: {data.get('usage', {})}")
                    print(f"Model used: {model}")
                
                end_time = time.time()
                print(f"Total request time: {end_time - start_time:.2f} seconds")
                return end_time - start_time
            else:
                print("Error:", await response.text())
                return None

async def test_rapid_requests(count=10, delay=0.1, model="GPT-4o-Mini"):
    """Test rapid requests to trigger rate limiting"""
    print(f"\nTesting {count} rapid requests with {delay}s delay between them using model: {model}...")
    
    # First check current key status
    initial_status = await test_current_key()
    if not initial_status:
        print("Failed to get initial key status")
        return
    
    # Store request times
    request_times = []
    successful_requests = 0
    failed_requests = 0
    
    # Make rapid requests
    for i in range(count):
        print(f"\nRequest {i+1}/{count}")
        prompt = f"Give me a one-sentence fact about the number {i+1}"
        request_time = await test_generate(
            model=model,
            prompt=prompt,
            stream=False
        )
        if request_time:
            request_times.append(request_time)
            successful_requests += 1
        else:
            failed_requests += 1
        
        # Check key status after each request
        await test_current_key()
        
        # Small delay between requests
        await asyncio.sleep(delay)
    
    # Final key status
    final_status = await test_current_key()
    
    # Print summary
    print("\nRapid Request Test Summary:")
    print(f"- Total requests: {count}")
    print(f"- Successful requests: {successful_requests}")
    print(f"- Failed requests: {failed_requests}")
    print(f"- Model used: {model}")
    
    if request_times:
        print(f"- Average request time: {sum(request_times)/len(request_times):.2f} seconds")
    else:
        print("- Average request time: N/A (all requests failed)")
    
    print(f"- Initial key: {initial_status['current_key']}")
    print(f"- Final key: {final_status['current_key']}")
    
    # Check if key switching occurred
    if initial_status['current_key'] != final_status['current_key']:
        print("✅ Key switching occurred during rapid requests")
    else:
        print("ℹ️ No key switching occurred during rapid requests")

async def test_key_switching():
    """Test manual key switching"""
    print("\nTesting manual key switching...")
    
    # Get initial key status
    initial_status = await test_current_key()
    if not initial_status:
        print("Failed to get initial key status")
        return
    
    # If fallback key is not available, we can't test switching
    if not initial_status['fallback_key_available']:
        print("❌ Fallback key not available, can't test key switching")
        return
    
    # Determine which key to switch to
    target_key = "fallback" if initial_status['current_key'] == "primary" else "primary"
    
    # Switch to the other key
    switch_result = await switch_key(target_key)
    if not switch_result:
        print("Failed to switch key")
        return
    
    # Verify the switch
    new_status = await test_current_key()
    if new_status['current_key'] == target_key:
        print(f"✅ Successfully switched to {target_key} key")
    else:
        print(f"❌ Failed to switch to {target_key} key")
    
    # Switch back to the original key
    await switch_key(initial_status['current_key'])
    final_status = await test_current_key()
    
    if final_status['current_key'] == initial_status['current_key']:
        print(f"✅ Successfully switched back to {initial_status['current_key']} key")
    else:
        print(f"❌ Failed to switch back to {initial_status['current_key']} key")

async def test_parallel_requests(count=5, model="GPT-4o-Mini"):
    """Test parallel requests to see how the system handles concurrent load"""
    print(f"\nTesting {count} parallel requests using model: {model}...")
    
    # First check current key status
    initial_status = await test_current_key()
    
    # Create tasks for parallel execution
    tasks = []
    for i in range(count):
        prompt = f"Give me a one-sentence fact about the number {i+1}"
        task = test_generate(
            model=model,
            prompt=prompt,
            stream=False
        )
        tasks.append(task)
    
    # Execute all tasks in parallel
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    # Check for exceptions
    success_count = 0
    error_count = 0
    for result in results:
        if isinstance(result, Exception):
            error_count += 1
        elif result is not None:
            success_count += 1
    
    # Final key status
    final_status = await test_current_key()
    
    # Print summary
    print("\nParallel Request Test Summary:")
    print(f"- Total requests: {count}")
    print(f"- Successful requests: {success_count}")
    print(f"- Failed requests: {error_count}")
    print(f"- Model used: {model}")
    print(f"- Total time for all requests: {end_time - start_time:.2f} seconds")
    print(f"- Initial key: {initial_status['current_key']}")
    print(f"- Final key: {final_status['current_key']}")
    
    # Check if key switching occurred
    if initial_status['current_key'] != final_status['current_key']:
        print("✅ Key switching occurred during parallel requests")
    else:
        print("ℹ️ No key switching occurred during parallel requests")

async def test_rate_limit_recovery(model="GPT-4o-Mini"):
    """Test how the system recovers from rate limiting"""
    print(f"\nTesting rate limit recovery using model: {model}...")
    
    # First check current key status
    initial_status = await test_current_key()
    
    # Make rapid requests to trigger rate limiting
    print("Making rapid requests to trigger rate limiting...")
    rapid_count = 5
    for i in range(rapid_count):
        prompt = f"Quick response {i+1}: Give me a one-word answer."
        await test_generate(
            model=model,
            prompt=prompt,
            stream=False
        )
        # Very small delay between requests
        await asyncio.sleep(0.1)
    
    # Check status after rapid requests
    after_rapid_status = await test_current_key()
    
    # Determine which key has a non-zero wait time (if any)
    primary_wait = after_rapid_status['wait_times']['primary']
    fallback_wait = after_rapid_status['wait_times'].get('fallback', 0) or 0
    
    # Find the maximum wait time
    recovery_time = max(primary_wait, fallback_wait)
    
    if recovery_time > 0:
        print(f"Waiting {recovery_time + 2} seconds for rate limit to recover...")
        await asyncio.sleep(recovery_time + 2)  # Add a buffer
    else:
        print("No wait time detected after rapid requests. Waiting 5 seconds anyway...")
        await asyncio.sleep(5)  # Wait a bit anyway to ensure recovery
    
    # Check status after recovery
    recovery_status = await test_current_key()
    
    # Make one more request to verify recovery
    print("\nMaking request after recovery period...")
    await test_generate(
        model=model,
        prompt="Has the rate limit recovered?",
        stream=False
    )
    
    # Final status
    final_status = await test_current_key()
    
    # Print summary
    print("\nRate Limit Recovery Test Summary:")
    print(f"- Model used: {model}")
    print(f"- Initial primary wait time: {initial_status['wait_times']['primary']} seconds")
    if initial_status['wait_times'].get('fallback') is not None:
        print(f"- Initial fallback wait time: {initial_status['wait_times']['fallback']} seconds")
    
    print(f"- After rapid requests primary wait time: {after_rapid_status['wait_times']['primary']} seconds")
    if after_rapid_status['wait_times'].get('fallback') is not None:
        print(f"- After rapid requests fallback wait time: {after_rapid_status['wait_times']['fallback']} seconds")
    
    print(f"- After recovery primary wait time: {recovery_status['wait_times']['primary']} seconds")
    if recovery_status['wait_times'].get('fallback') is not None:
        print(f"- After recovery fallback wait time: {recovery_status['wait_times']['fallback']} seconds")
    
    # Check if recovery was successful - modified logic to handle key switching
    recovery_successful = False
    
    # Case 1: If both keys had wait times that decreased
    if (primary_wait > 0 and recovery_status['wait_times']['primary'] < primary_wait) or \
       (fallback_wait > 0 and recovery_status['wait_times'].get('fallback', 0) < fallback_wait):
        recovery_successful = True
    
    # Case 2: If we switched keys and now both have low wait times
    if after_rapid_status['current_key'] != initial_status['current_key'] and \
       recovery_status['wait_times']['primary'] < 1 and \
       (recovery_status['wait_times'].get('fallback', 0) or 0) < 1:
        recovery_successful = True
    
    # Case 3: If wait times were already low after rapid requests (key switching worked well)
    if primary_wait < 1 and fallback_wait < 1:
        recovery_successful = True
    
    if recovery_successful:
        print("✅ Rate limit recovery successful (or key switching handled the load well)")
    else:
        print("❌ Rate limit did not recover as expected")

async def test_alternating_requests(count=6, delay=0.5, model="GPT-4o-Mini"):
    """Test alternating between different endpoints to ensure all use the rate limiter"""
    print(f"\nTesting {count} alternating requests with {delay}s delay between them using model: {model}...")
    
    # First check current key status
    initial_status = await test_current_key()
    if not initial_status:
        print("Failed to get initial key status")
        return
    
    # Store request times
    request_times = []
    successful_requests = 0
    failed_requests = 0
    
    # Make alternating requests
    for i in range(count):
        print(f"\nRequest {i+1}/{count}")
        
        # Alternate between different endpoints
        if i % 3 == 0:
            # Use generate endpoint
            prompt = f"Give me a one-sentence fact about the number {i+1}"
            request_time = await test_generate(
                model=model,
                prompt=prompt,
                stream=False
            )
        elif i % 3 == 1:
            # Use chat completions endpoint
            request_time = await test_chat_completions(stream=False, model=model)
        else:
            # Use chat completions with streaming
            request_time = await test_chat_completions(stream=True, model=model)
        
        if request_time:
            request_times.append(request_time)
            successful_requests += 1
        else:
            failed_requests += 1
        
        # Check key status after each request
        await test_current_key()
        
        # Small delay between requests
        await asyncio.sleep(delay)
    
    # Final key status
    final_status = await test_current_key()
    
    # Print summary
    print("\nAlternating Request Test Summary:")
    print(f"- Total requests: {count}")
    print(f"- Successful requests: {successful_requests}")
    print(f"- Failed requests: {failed_requests}")
    print(f"- Model used: {model}")
    
    if request_times:
        print(f"- Average request time: {sum(request_times)/len(request_times):.2f} seconds")
    else:
        print("- Average request time: N/A (all requests failed)")
    
    print(f"- Initial key: {initial_status['current_key']}")
    print(f"- Final key: {final_status['current_key']}")

async def main():
    try:
        print("Starting Dual-Key Rate Limiting Tests")
        print("=====================================")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            test_name = sys.argv[1]
            if test_name == "tags":
                await test_tags()
            elif test_name == "current_key":
                await test_current_key()
            elif test_name == "switch_key":
                key_type = sys.argv[2] if len(sys.argv) > 2 else "primary"
                await switch_key(key_type)
            elif test_name == "generate":
                model = sys.argv[2] if len(sys.argv) > 2 else "GPT-4o-Mini"
                stream = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
                await test_generate(model=model, stream=stream)
            elif test_name == "chat":
                stream = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
                model = sys.argv[3] if len(sys.argv) > 3 else "GPT-4o-Mini"
                await test_chat_completions(stream=stream, model=model)
            elif test_name == "rapid":
                count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
                model = sys.argv[4] if len(sys.argv) > 4 else "GPT-4o-Mini"
                await test_rapid_requests(count=count, delay=delay, model=model)
            elif test_name == "alternating":
                count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
                delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
                model = sys.argv[4] if len(sys.argv) > 4 else "GPT-4o-Mini"
                await test_alternating_requests(count=count, delay=delay, model=model)
            elif test_name == "parallel":
                count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
                model = sys.argv[3] if len(sys.argv) > 3 else "GPT-4o-Mini"
                await test_parallel_requests(count=count, model=model)
            elif test_name == "recovery":
                model = sys.argv[2] if len(sys.argv) > 2 else "GPT-4o-Mini"
                await test_rate_limit_recovery(model=model)
            elif test_name == "key_switching":
                await test_key_switching()
            elif test_name == "claude-test":
                # Special test for Claude-3.7-Sonnet rate limiting
                print("\nRunning Claude-3.7-Sonnet rate limit tests...")
                
                # First test basic connectivity with GPT-4o-Mini
                await test_generate(model="GPT-4o-Mini", prompt="Quick test with GPT-4o-Mini")
                
                # Test chat completions with Claude-3.7-Sonnet
                await test_chat_completions(stream=False, model="Claude-3.7-Sonnet")
                
                # Then run rate limit tests with Claude-3.7-Sonnet
                await test_rapid_requests(count=5, delay=0.5, model="Claude-3.7-Sonnet")
                await test_rate_limit_recovery(model="Claude-3.7-Sonnet")
                await test_parallel_requests(count=3, model="Claude-3.7-Sonnet")
            else:
                print(f"Unknown test: {test_name}")
                print("Available tests: tags, current_key, switch_key, generate, chat, rapid, alternating, parallel, recovery, key_switching, claude-test")
        else:
            # Run all tests in sequence with default models
            
            # Basic endpoint tests
            await test_tags()
            await test_current_key()
            
            # Test key switching
            await test_key_switching()
            
            # Test single requests
            await test_generate(stream=False)
            await test_chat_completions(stream=False)
            
            # Test rapid requests to trigger rate limiting
            await test_rapid_requests(count=5, delay=0.5)
            
            # Test alternating between different endpoints
            await test_alternating_requests(count=6, delay=0.5)
            
            # Test parallel requests
            await test_parallel_requests(count=3)
            
            # Test rate limit recovery
            await test_rate_limit_recovery()
            
    except aiohttp.ClientError as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 