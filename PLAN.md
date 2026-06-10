Here's a comprehensive plan for your llama.cpp benchmarking script:

---

## Benchmark Script Plan

### Architecture Overview

```
llama_bench.py
├── Config (server URL, timeouts, model params)
├── LlamaCppClient (HTTP interface to /v1/chat/completions or /completion)
├── BenchmarkResult (dataclass: pass/fail + timing metrics)
├── Exercise suite (20+ tasks with validators)
└── Reporter (summary table + JSON export)
```

---

### Metrics Captured

The llama.cpp server returns `timings` in every response body. The client will extract:

| Metric | Source |
|---|---|
| `prompt_eval_time_ms` | `timings.prompt_eval_time` |
| `prompt_tokens` | `timings.prompt_eval_n` |
| `prompt_speed_tok_s` | derived: tokens / time |
| `gen_time_ms` | `timings.eval_time` |
| `gen_tokens` | `timings.eval_n` |
| `gen_speed_tok_s` | derived: tokens / time |
| `total_wall_time_ms` | measured with `time.perf_counter` around the full HTTP call |
| `ttft_ms` | Time-to-first-token (streaming mode only) |

---

### Exercise Suite (25 tasks across 6 categories)

**Exact Match — Math & Logic (6)**
1. Single-digit arithmetic: `"What is 7 + 8?"` → `15`
2. Multi-step arithmetic: `"What is (12 * 4) - 17?"` → `31`
3. Modulo: `"What is 100 mod 7?"` → `2`
4. Prime check: `"Is 97 a prime number? Answer yes or no."` → `yes`
5. Fibonacci: `"What is the 10th Fibonacci number?"` → `55`
6. Roman numeral: `"Convert XLII to a decimal number."` → `42`

**Exact Match — Factual (5)**

7. Country capital: `"What is the capital of Japan?"` → `Tokyo`
8. Element symbol: `"What is the chemical symbol for gold?"` → `Au`
9. Planet count: `"How many planets are in our solar system?"` → `8`
10. Speed of light: `"What is the speed of light in m/s? Give just the number."` → `299792458`
11. Boiling point: `"At what temperature (Celsius) does water boil at sea level?"` → `100`

**Pattern / Contains Match — Reasoning (5)**

12. Odd one out: `"Which doesn't belong: apple, banana, carrot, grape?"` → validator checks for `carrot`
13. Analogy: `"Hot is to cold as day is to ___?"` → validator checks for `night`
14. Next in sequence: `"2, 4, 8, 16, ___"` → `32`
15. Rhyme: `"Give one word that rhymes with 'cat'."` → validator checks it ends in `-at`
16. Acronym expansion: `"What does CPU stand for?"` → validator checks for `central processing unit` (case-insensitive)

**Code Generation (4)**

17. FizzBuzz snippet: ask for Python FizzBuzz 1–15, validate output contains `FizzBuzz` at position 15
18. Reverse a string: ask for a Python one-liner, run `eval()` against a test input
19. List comprehension: squares of 1–5, validate `[1, 4, 9, 16, 25]` appears in response
20. JSON parse: ask model to write code that parses a given JSON string, validate syntactically

**Instruction Following (3)**

21. Word count constraint: `"List exactly 3 fruits, one per line."` → validator counts lines = 3
22. Format compliance: `"Respond with only a valid JSON object with keys 'name' and 'age'."` → validate `json.loads()`
23. Negative constraint: `"Describe a cat without using the word 'cat'."` → validate `cat` not in response

**Long Context / Summarization (2)**

24. Summarization length: provide a 200-word paragraph, ask for a summary under 50 words → validate word count ≤ 50
25. Extraction: embed a fake "secret code" in a paragraph of filler text, ask model to find it → exact match

---

### Validator Design

Each exercise is a dataclass:

```python
@dataclass
class Exercise:
    id: int
    category: str
    prompt: str
    validator: Callable[[str], bool]
    description: str
```

Validators are pure functions `(response: str) -> bool`. Types used:
- `exact_match(expected)` — strips/lowercases both sides
- `contains(substring)` — case-insensitive
- `regex_match(pattern)`
- `json_valid()` — wraps `json.loads`
- `word_count_lte(n)`
- `python_eval(code_template, expected_output)` — extracts code block, runs in subprocess sandbox

---

### Execution Flow

```
1. Parse CLI args (--host, --port, --model, --repeat, --output, --stream)
2. Health-check: GET /health or a ping completion
3. For each exercise (optionally repeated N times):
   a. Record wall-clock start
   b. POST to /v1/chat/completions (or /completion)
   c. Record wall-clock end
   d. Extract timings{} from response body
   e. Run validator → pass/fail
   f. Store BenchmarkResult
4. Aggregate: pass rate, median/p95 speeds per category
5. Print rich summary table
6. Export results to JSON (optional CSV)
```

---

### CLI Interface

```
python llama_bench.py \
  --host localhost \
  --port 8080 \
  --repeat 3 \           # run each exercise N times, report median
  --categories math,code \  # optional filter
  --output results.json \
  --stream               # use streaming to capture TTFT
  --verbose              # print each response
```

---

### Output

**Console:**
```
┌─────────────────────────────────────────────────────────────────┐
│  llama.cpp Benchmark  |  model: mistral-7b  |  2026-06-09       │
├──────┬──────────────────┬────────┬──────────┬────────┬──────────┤
│  #   │ Description      │ Result │ PP tok/s │ TG tok/s│ Wall ms │
├──────┼──────────────────┼────────┼──────────┼────────┼──────────┤
│  1   │ 7 + 8            │  PASS  │  1842    │  48.3  │  312    │
│  2   │ (12*4)-17        │  PASS  │  1790    │  51.1  │  289    │
│ ...  │                  │        │          │        │         │
├──────┴──────────────────┴────────┴──────────┴────────┴──────────┤
│  SCORE: 22/25 (88%)   Median PP: 1821 tok/s   Median TG: 49.7  │
└─────────────────────────────────────────────────────────────────┘
```

**JSON export** includes raw per-exercise results, all timing fields, and aggregate stats per category.

---

### Dependencies

```
requests       # HTTP calls
rich           # console table formatting
argparse       # CLI
dataclasses    # result types
subprocess     # sandboxed code eval
json, re, time # stdlib
```

No llama.cpp-specific SDK needed — it just speaks to the OpenAI-compatible REST endpoint.
