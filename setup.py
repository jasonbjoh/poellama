from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define fallback requirements in case requirements.txt is not available
fallback_requirements = [
    "fastapi>=0.68.0,<1.0.0",
    "uvicorn>=0.15.0,<1.0.0",
    "python-dotenv>=0.19.0,<1.0.0",
    "fastapi-poe>=0.0.16,<1.0.0",
    "pydantic>=1.8.0,<2.0.0",
    "python-multipart>=0.0.5,<1.0.0",
    "aiohttp>=3.8.0,<4.0.0",
    "typing-extensions>=4.0.0,<5.0.0"
]

# Try to read from requirements.txt, fall back to hardcoded list if not available
try:
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    else:
        requirements = fallback_requirements
except:
    requirements = fallback_requirements

setup(
    name="poellama",
    version="0.1.0",
    author="jasonbjoh",
    author_email="",  # Email omitted for privacy
    description="An Ollama API wrapper for Poe.com AI - Use Poe's models with Ollama-compatible applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasonbjoh/poellama",
    packages=find_packages(),
    package_data={
        "poellama": ["models.json"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "poellama=poellama.main:run_server",
        ],
    },
)
