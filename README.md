# Python Llama Benchmarking Script

A python script to help benchmark local AI models.

Coded with assistance from Claude (Sonnet 4.6, Opus 4.6).

## Initial prompt:
> Generate a plan for a python script that will help benchmark local AI models running llama.cpp. I'll be running the server and want to capture things like prompt-processing and token generation speeds, as well as total execution time. I want the script to have a series of exercises, at least 20, that it runs through with verifiable results so it can output success metrics, like 14/20 pass or something like that.

This took a few tries because I use the --models-preset flag when running llama.cpp so it acts a bit more like a router.

Use `uv sync` to get the package requirements.   
Can activate with `source .venv/bin/activate`

### Discover models to use
`python llama_bench.py --list-models`

### Basic run (server on localhost:8080)
`python llama_bench.py --model mistral-7b`

### Benchmark an Ollama server (defaults to localhost:11434)
`python llama_bench.py --ollama --model llama3`

The `--ollama` flag activates the Ollama endpoints (`/api/chat`) and defaults
the port to `11434`. Without it, the llama.cpp endpoints are used.

### Filter to just math and code, run each 3 times (reports median)
`python llama_bench.py --model llama3 --categories Math,Code --repeat 3`

### Save results to JSON, verbose mode
`python llama_bench.py --model phi3 --output results.json --verbose`

### List all exercises
`python llama_bench.py --list`

### Get full error message
`python llama_bench.py --diagnose`

## Executions
```
python llama_bench.py --model gemma4:12b-q3 --output data/gemma4-12b-q3.json --repeat 3
python llama_bench.py --model qwen3.5:9b --output data/qwen3.5-9b.json --repeat 3
python llama_bench.py --model qwen3:8b --output data/qwen3-8b.json --repeat 3
python llama_bench.py --model llama3.3:8b --output data/llama3.3-8b.json --repeat 3
python llama_bench.py --model gemma4:26b-moe --output data/gemma4-26b-moe.json --repeat 3

# With a different version of llama.cpp (not working at the moment):
python llama_bench.py --model bonsai:8b --output data/bonsai-8b.json --repeat 3

python bench_report.py

# Mac tests
python llama_bench.py --ollama --model gemma4:12b-mlx --output data/gemma4-12b-mlx.json --repeat 3
python llama_bench.py --ollama --model qwen3.6:27b-coding-nvfp4 --output data/qwen3.6-27b-coding-nvfp4.json --repeat 3
python llama_bench.py --ollama --model gemma4:26b-nvfp4 --output data/gemma4-26b-nvfp4.json --repeat 3
python llama_bench.py --ollama --model qwen3.6:35b-a3b-nvfp4 --output data/qwen3.6-35b-a3b-nvfp4.json --repeat 3
```
