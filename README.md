# Python Llama Benchmarking Script

A python script to help benchmark local AI models.

Coded with assistance from Claude (Sonnet 4.6).

Initial prompt:
> Generate a plan for a python script that will help benchmark local AI models running llama.cpp. I'll be running the server and want to capture things like prompt-processing and token generation speeds, as well as total execution time. I want the script to have a series of exercises, at least 20, that it runs through with verifiable results so it can output success metrics, like 14/20 pass or something like that.

Use `uv sync` to get the package requirements

# Basic run (server on localhost:8080)
`python llama_bench.py --model mistral-7b`

# Filter to just math and code, run each 3 times (reports median)
`python llama_bench.py --model llama3 --categories Math,Code --repeat 3`

# Save results to JSON, verbose mode
`python llama_bench.py --model phi3 --output results.json --verbose`

# List all exercises
`python llama_bench.py --list`

