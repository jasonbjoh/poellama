import aiohttp
import asyncio
import json

async def test_models_endpoint():
    print("\nTesting /models endpoint...")
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/models') as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print("Available models:")
            for model in data['models']:
                print(f"- {model['name']}: {model['cost']} (Context: {model['contextWindow']})")

async def test_chat_endpoint():
    print("\nTesting /chat endpoint...")
    test_messages = [
        {
            "model": "Claude-3.5-Haiku",
            "message": "What is the capital of France?"
        },
        {
            "model": "GPT-3.5-Turbo",
            "message": "Write a haiku about programming."
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for test in test_messages:
            print(f"\nTesting with model: {test['model']}")
            print(f"Message: {test['message']}")
            
            async with session.post(
                'http://localhost:8000/chat',
                params={'model': test['model'], 'message': test['message']}
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print("Response:")
                    print(data['response'])
                else:
                    print("Error:", await response.text())
            
            # Wait a bit between requests to respect rate limiting
            await asyncio.sleep(2)

async def main():
    try:
        print("Starting Poe API Tests")
        print("====================")
        
        # Test /models endpoint
        await test_models_endpoint()
        
        # Test /chat endpoint
        await test_chat_endpoint()
        
    except aiohttp.ClientError as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 