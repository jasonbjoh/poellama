# PoeLlama

An Ollama API wrapper for Poe.com AI. This project allows you to use Poe's powerful AI models through Ollama-compatible applications and tools. It provides a FastAPI server that translates Ollama API calls to Poe API calls, making it seamless to use Poe's models in your existing Ollama workflows.

## Features

- **Full Ollama API Compatibility**
  - `/api/generate` endpoint for text generation
  - `/api/tags` endpoint for model listing
  - Streaming support for real-time responses
- **Additional OpenAI Compatibility**
  - `/v1/chat/completions` endpoint for OpenAI-style interactions
- **Original Poe Endpoints**
  - Simple `/chat` endpoint for direct Poe API access
  - `/models` endpoint for complete model information
- **Advanced Features**
  - Rate limiting to prevent API overuse
  - Detailed logging for debugging
  - Environment variable configuration

## Installation

```bash
pip install poellama
```

Or install from source:

```bash
git clone https://github.com/yourusername/poellama
cd poellama
pip install -e .
```

## Configuration

1. Create a `.env` file in your project directory:
```env
POE_API_KEY=your_poe_api_key_here
```

2. (Optional) Configure logging level in `main.py`:
```python
setup_logging(debug_level=DebugLevel.VERBOSE)  # or DebugLevel.MINIMAL
```

## Usage

### Starting the Server

```bash
poellama
```

Or run directly with Python:

```bash
python -m poellama.main
```

The server will start on `http://localhost:8000`

### Using with Ollama Tools

You can use any Ollama-compatible tool by pointing it to your PoeLlama server:

```python
import requests

# Generate text
response = requests.post("http://localhost:8000/api/generate", json={
    "model": "Claude-3.5-Haiku",
    "prompt": "Hello, how are you?",
    "stream": False
})

# List available models
response = requests.get("http://localhost:8000/api/tags")
```

### OpenAI Compatibility Layer

For applications that use the OpenAI SDK:

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"  # any string will work

response = openai.ChatCompletion.create(
    model="Claude-3.5-Haiku",  # or any other supported model
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    stream=True  # optional
)
```

### Direct Poe API Access

For simple interactions:

```python
import requests

response = requests.post("http://localhost:8000/chat", params={
    "model": "Claude-3.5-Haiku",
    "message": "Hello, how are you?"
})
```

## Available Models

- GPT-3.5-Turbo
- GPT-4o-Mini
- GPT-4o
- Claude-3.5-Sonnet
- Claude-3.5-Haiku
- Gemini-2.0-Flash-Lite

## Development

### Running Tests

```bash
python test_server.py  # Test basic functionality
python test_ollama.py  # Test Ollama compatibility
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for the API specification
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Poe API](https://poe.com) for the underlying AI services 