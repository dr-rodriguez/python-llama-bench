#!/usr/bin/env python3
"""
llama_bench.py — Benchmark script for llama.cpp or Ollama servers
Measures prompt-processing speed, token generation speed, TTFT, and wall time
across 25 exercises with verifiable results.
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimingMetrics:
    prompt_tokens: int = 0
    prompt_eval_time_ms: float = 0.0
    prompt_speed_tok_s: float = 0.0
    gen_tokens: int = 0
    gen_time_ms: float = 0.0
    gen_speed_tok_s: float = 0.0
    total_wall_time_ms: float = 0.0
    ttft_ms: float = 0.0


@dataclass
class BenchmarkResult:
    exercise_id: int
    category: str
    description: str
    passed: bool
    response_text: str
    metrics: TimingMetrics
    error: Optional[str] = None


@dataclass
class Exercise:
    id: int
    category: str
    description: str
    prompt: str
    validator: Callable[[str], bool]
    system_prompt: str = "You are a helpful assistant. Be concise and direct."
    max_tokens: int = 300


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def exact_match(*expected_variants: str) -> Callable[[str], bool]:
    """Pass if response (stripped, lowercased) contains any expected variant."""
    def _v(resp: str) -> bool:
        r = resp.strip().lower()
        return any(e.lower() in r for e in expected_variants)
    return _v


def starts_with_number(expected: int) -> Callable[[str], bool]:
    """Pass if response starts with or contains the expected number."""
    def _v(resp: str) -> bool:
        nums = re.findall(r'\b\d+\b', resp)
        return str(expected) in nums
    return _v


def contains_all(*substrings: str) -> Callable[[str], bool]:
    """Pass if response contains all substrings (case-insensitive)."""
    def _v(resp: str) -> bool:
        r = resp.lower()
        return all(s.lower() in r for s in substrings)
    return _v


def contains_any(*substrings: str) -> Callable[[str], bool]:
    """Pass if response contains any of the substrings (case-insensitive)."""
    def _v(resp: str) -> bool:
        r = resp.lower()
        return any(s.lower() in r for s in substrings)
    return _v


def regex_match(pattern: str, flags=re.IGNORECASE) -> Callable[[str], bool]:
    def _v(resp: str) -> bool:
        return bool(re.search(pattern, resp, flags))
    return _v


def json_valid_with_keys(*keys: str) -> Callable[[str], bool]:
    """Pass if response contains valid JSON with all specified keys."""
    def _v(resp: str) -> bool:
        # Try to extract JSON from markdown code block or raw
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp, re.DOTALL)
        candidate = json_match.group(1) if json_match else resp.strip()
        # Strip any leading/trailing non-JSON text
        brace_match = re.search(r'\{.*\}', candidate, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
        try:
            data = json.loads(candidate)
            return all(k in data for k in keys)
        except (json.JSONDecodeError, ValueError):
            return False
    return _v


def word_count_lte(max_words: int) -> Callable[[str], bool]:
    def _v(resp: str) -> bool:
        return len(resp.split()) <= max_words
    return _v


def line_count_equals(expected: int) -> Callable[[str], bool]:
    """Pass if response has exactly N non-empty lines."""
    def _v(resp: str) -> bool:
        lines = [l.strip() for l in resp.strip().splitlines() if l.strip()]
        return len(lines) == expected
    return _v


def not_contains(substring: str) -> Callable[[str], bool]:
    def _v(resp: str) -> bool:
        return substring.lower() not in resp.lower()
    return _v


def python_code_produces(expected_output: str) -> Callable[[str], bool]:
    """Extract Python code from response, run it, check output matches expected."""
    def _v(resp: str) -> bool:
        # Extract code block
        code_match = re.search(r'```(?:python)?\s*(.*?)```', resp, re.DOTALL)
        code = code_match.group(1).strip() if code_match else resp.strip()
        # Remove any non-code preamble lines (heuristic: skip until first def/import/for/print)
        lines = code.splitlines()
        start = 0
        for i, line in enumerate(lines):
            if re.match(r'^\s*(def |import |for |print|[a-zA-Z_])', line):
                start = i
                break
        code = "\n".join(lines[start:])
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=5
            )
            return expected_output.strip() in result.stdout.strip()
        except Exception:
            return False
    return _v


def contains_secret(secret: str) -> Callable[[str], bool]:
    def _v(resp: str) -> bool:
        return secret.upper() in resp.upper()
    return _v


def ends_with_pattern(suffix_pattern: str) -> Callable[[str], bool]:
    """Pass if any word in the response ends with the given suffix."""
    def _v(resp: str) -> bool:
        words = re.findall(r'\b\w+\b', resp.lower())
        return any(w.endswith(suffix_pattern.lower()) for w in words)
    return _v


# ---------------------------------------------------------------------------
# Exercise suite
# ---------------------------------------------------------------------------

FILLER_TEXT = (
    "The conference began on a Tuesday with opening remarks from the director. "
    "Attendees gathered in the main hall to discuss quarterly results. "
    "The secret code is ZETA9. "
    "Following the keynote, breakout sessions were held in adjacent rooms. "
    "Participants exchanged business cards and discussed future collaborations. "
    "The day ended with a networking dinner near the waterfront."
)

SUMMARY_TEXT = (
    "Artificial intelligence has made remarkable strides in recent years, transforming "
    "industries from healthcare to finance. Machine learning models can now diagnose "
    "diseases, generate realistic images, write coherent text, and even compose music. "
    "Despite these advances, significant challenges remain around safety, bias, and "
    "the energy costs of training large models. Researchers are actively working on "
    "more efficient architectures and better alignment techniques to ensure AI systems "
    "behave as intended. The coming decade will likely see continued rapid progress, "
    "with AI becoming an ever more integral part of daily life and professional work."
)

def build_exercises() -> list[Exercise]:
    return [
        # ── Math & Logic ─────────────────────────────────────────────────────
        Exercise(
            id=1, category="Math",
            description="Single-digit addition (7 + 8)",
            prompt="What is 7 + 8? Reply with only the number.",
            validator=starts_with_number(15),
        ),
        Exercise(
            id=2, category="Math",
            description="Multi-step arithmetic ((12 * 4) - 17)",
            prompt="What is (12 * 4) - 17? Reply with only the number.",
            validator=starts_with_number(31),
        ),
        Exercise(
            id=3, category="Math",
            description="Modulo (100 mod 7)",
            prompt="What is 100 mod 7? Reply with only the number.",
            validator=starts_with_number(2),
        ),
        Exercise(
            id=4, category="Math",
            description="Prime check (97)",
            prompt="Is 97 a prime number? Reply with only 'yes' or 'no'.",
            validator=exact_match("yes"),
        ),
        Exercise(
            id=5, category="Math",
            description="10th Fibonacci number",
            prompt="What is the 10th Fibonacci number (starting 1, 1, 2, 3, ...)? Reply with only the number.",
            validator=starts_with_number(55),
        ),
        Exercise(
            id=6, category="Math",
            description="Roman numeral XLII to decimal",
            prompt="Convert the Roman numeral XLII to a decimal number. Reply with only the number.",
            validator=starts_with_number(42),
        ),

        # ── Factual Knowledge ────────────────────────────────────────────────
        Exercise(
            id=7, category="Factual",
            description="Capital of Myanmar",
            prompt="What is the capital of Myanmar? Reply with only the city name.",
            validator=exact_match("Naypyidaw", "Nay Pyi Taw"),
        ),
        Exercise(
            id=8, category="Factual",
            description="Chemical symbol for tungsten",
            prompt="What is the chemical symbol for tungsten? Reply with only the symbol.",
            validator=exact_match("W"),
        ),
        Exercise(
            id=9, category="Factual",
            description="Number of bones in the adult human body",
            prompt="How many bones are in the adult human body? Reply with only the number.",
            validator=starts_with_number(206),
        ),
        Exercise(
            id=10, category="Factual",
            description="Year the Berlin Wall fell",
            prompt="In what year did the Berlin Wall fall? Reply with only the four-digit year.",
            validator=starts_with_number(1989),
        ),
        Exercise(
            id=11, category="Factual",
            description="Deepest ocean trench",
            prompt="What is the name of the deepest ocean trench on Earth? Reply with only the name.",
            validator=contains_any("mariana", "marianas"),
        ),

        # ── Reasoning & Language ─────────────────────────────────────────────
        Exercise(
            id=12, category="Reasoning",
            description="Odd one out (abstract: river among geometry)",
            prompt="Which word does not belong with the others? parallelogram, rhombus, river, trapezoid. Reply with only the word.",
            validator=exact_match("river"),
            max_tokens=1024,
        ),
        Exercise(
            id=13, category="Reasoning",
            description="Letter sequence: O, T, T, F, F, S, S, ___",
            prompt="What is the next letter in the sequence: O, T, T, F, F, S, S, ___? (Hint: think about counting.) Reply with only the single letter.",
            validator=exact_match("E"),
            max_tokens=1024,
        ),
        Exercise(
            id=14, category="Reasoning",
            description="Next in sequence: 1, 1, 2, 3, 5, 8, 13, ___",
            prompt="What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ___? Reply with only the number.",
            validator=starts_with_number(21),
            max_tokens=1024,
        ),
        Exercise(
            id=15, category="Reasoning",
            description="Counterfactual: days in February in a leap year",
            prompt="If a year is divisible by 400, how many days does February have in that year? Reply with only the number.",
            validator=starts_with_number(29),
            max_tokens=1024,
        ),
        Exercise(
            id=16, category="Reasoning",
            description="Verbal reasoning: before yesterday's tomorrow",
            prompt="If today is Wednesday, what day was 'the day before yesterday's tomorrow'? Think step by step, then reply with only the day name.",
            validator=exact_match("Tuesday"),
            max_tokens=1024,
        ),

        # ── Instruction Following ─────────────────────────────────────────────
        Exercise(
            id=17, category="Instruction",
            description="List exactly 3 fruits, one per line",
            prompt="List exactly 3 fruits. Put each fruit on its own line. Do not number them or add any other text.",
            validator=line_count_equals(3),
        ),
        Exercise(
            id=18, category="Instruction",
            description="Respond with valid JSON {name, age}",
            prompt='Respond with only a valid JSON object containing exactly two keys: "name" (any string value) and "age" (any integer value). No other text.',
            validator=json_valid_with_keys("name", "age"),
            max_tokens=1024,
        ),
        Exercise(
            id=19, category="Instruction",
            description="Describe a cat without using the word 'cat'",
            prompt="Describe a common household pet that meows and purrs in one sentence. Do not use the word 'cat' anywhere in your response.",
            validator=not_contains("cat"),
        ),
        Exercise(
            id=20, category="Instruction",
            description="Answer in exactly one word",
            prompt="What is the opposite of 'hot'? Reply with exactly one word.",
            validator=regex_match(r'^\s*\w+\s*$'),
            max_tokens=1024,
        ),

        # ── Code Generation ───────────────────────────────────────────────────
        Exercise(
            id=21, category="Code",
            description="Python: squares list comprehension",
            prompt="Write a single Python print statement using a list comprehension that prints the squares of 1 through 5. Provide only the code, no explanation.",
            validator=python_code_produces("[1, 4, 9, 16, 25]"),
            max_tokens=1024,
        ),
        Exercise(
            id=22, category="Code",
            description="Python: reverse a string",
            prompt='Write a single Python print statement that prints the reverse of the string "hello". Provide only the code, no explanation.',
            validator=python_code_produces("olleh"),
            max_tokens=1024,
        ),
        Exercise(
            id=23, category="Code",
            description="Python: FizzBuzz output for n=15",
            prompt="Write a Python snippet that prints FizzBuzz for numbers 1 through 15 (print Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both). Provide only the code.",
            validator=python_code_produces("FizzBuzz"),
            max_tokens=1024,
        ),
        Exercise(
            id=24, category="Code",
            description="Python: sum of a list",
            prompt="Write a single Python print statement that prints the sum of the list [10, 20, 30, 40]. Provide only the code, no explanation.",
            validator=python_code_produces("100"),
            max_tokens=1024,
        ),

        # ── Long Context / Extraction ─────────────────────────────────────────
        Exercise(
            id=25, category="Extraction",
            description="Extract secret code from paragraph",
            prompt=(
                f"Read the following text and extract the secret code. "
                f"Reply with only the secret code, nothing else.\n\n{FILLER_TEXT}"
            ),
            validator=contains_secret("ZETA9"),
        ),
        Exercise(
            id=26, category="Extraction",
            description="Summarize text in ≤ 40 words",
            prompt=(
                f"Summarize the following text in no more than 40 words. "
                f"Be concise.\n\n{SUMMARY_TEXT}"
            ),
            validator=word_count_lte(40),
            max_tokens=1024,
        ),
    ]


# ---------------------------------------------------------------------------
# llama.cpp HTTP client
# ---------------------------------------------------------------------------

_ENDPOINT_CHAT    = "chat"     # POST /v1/chat/completions  (OpenAI-compat)
_ENDPOINT_NATIVE  = "native"   # POST /completion           (llama.cpp native)
_ENDPOINT_OLLAMA  = "ollama"   # POST /api/chat             (Ollama native)

_SERVER_LLAMACPP = "llamacpp"
_SERVER_OLLAMA   = "ollama"

# Stop sequences for the native /completion path so the model can't run past
# its answer and start hallucinating extra "### User:/### Assistant:" turns.
_NATIVE_STOP = ["### User:", "### System:", "### Assistant:", "</s>",
                "<|im_end|>", "<|eot_id|>", "<|end|>"]

# Reasoning models represent chain-of-thought differently. We only score the
# final answer, so strip whichever form a given model uses:
#   • XML-ish tags:  <think>...</think>, <thinking>...</thinking>   (Qwen, etc.)
#   • Harmony channels: <|channel|>analysis<|message|>...<|channel|>final<|message|>ANSWER  (gpt-oss)
# Note: a model run through the raw /completion endpoint (no chat template) may
# emit *untagged* reasoning that cannot be stripped reliably — the fix for that
# is using the chat endpoint, not this function.
_THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|(?:end|return)\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def strip_thinking(text: str) -> str:
    """Remove reasoning/thinking blocks, keeping only the final answer text."""
    if not text:
        return text

    # Harmony format: keep only the content of the `final` channel.
    if "<|channel|>" in text.lower():
        finals = _HARMONY_FINAL_RE.findall(text)
        if finals:
            text = finals[-1]
        else:
            # Only an (unfinished) analysis channel — strip the control tokens.
            text = _HARMONY_TOKEN_RE.sub("", text)

    cleaned = _THINK_RE.sub("", text)
    # Unclosed block (model hit the token cap mid-thought): drop from the tag on.
    lower = cleaned.lower()
    for tag in ("<think>", "<thinking>"):
        idx = lower.rfind(tag)
        if idx != -1 and ("</" + tag[1:]) not in lower[idx:]:
            cleaned = cleaned[:idx]
            lower = cleaned.lower()
    return cleaned.strip()


class LlamaCppClient:
    def __init__(
        self,
        host: str,
        port: int,
        model: str = "",
        timeout: int = 240,  # longer timeout for larger models
        diagnose: bool = False,
        ollama: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"
        self.model = model          # passed through in every request payload
        self.timeout = timeout
        self.diagnose = diagnose
        self.ollama = ollama        # force the Ollama pathway (skip auto-detection)
        self.session = requests.Session()
        self._endpoint_mode: Optional[str] = None   # set by probe()
        self._server_type: str = _SERVER_OLLAMA if ollama else _SERVER_LLAMACPP

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Query /v1/models (llama.cpp/OpenAI) or /api/tags (Ollama) for available model IDs."""
        try:
            r = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [m.get("id", "") for m in data.get("data", [])]
        except requests.RequestException:
            pass
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except requests.RequestException:
            pass
        return []

    # ------------------------------------------------------------------
    # Probe: figure out which endpoint + payload shape this server accepts
    # ------------------------------------------------------------------
    def probe(self) -> bool:
        """
        Try /health, then attempt a one-token completion on each known endpoint.
        Includes the model name in the payload when one is set — required by
        router-mode servers started with --models-preset.
        Sets self._endpoint_mode and returns True if the server is usable.
        """
        # 1. Use the Ollama pathway when explicitly requested via --ollama. Ollama
        #    also serves /v1/chat/completions but without timings, so we prefer its
        #    native /api/chat endpoint.
        if self.ollama:
            ollama_payload: dict = {
                "model": self.model or "",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            }
            try:
                r = self.session.post(
                    f"{self.base_url}/api/chat",
                    json=ollama_payload, timeout=15,
                )
                if r.status_code == 200:
                    self._endpoint_mode = _ENDPOINT_OLLAMA
                    self._server_type = _SERVER_OLLAMA
                    model_label = f" (model: {self.model})" if self.model else ""
                    console.print(f"[green]  Endpoint:[/green] /api/chat (Ollama){model_label} ✓")
                    return True
                else:
                    if self.diagnose:
                        console.print(f"[dim]  /api/chat → {r.status_code}: {r.text[:400]}[/dim]")
            except requests.RequestException as e:
                if self.diagnose:
                    console.print(f"[dim]  /api/chat error: {e}[/dim]")
            if not self.model:
                available = self.list_models()
                if available:
                    console.print(
                        f"[yellow]  Hint:[/yellow] --ollama set but the /api/chat probe failed.\n"
                        f"  Available models: {', '.join(available)}\n"
                        f"  Re-run with e.g. [dim]--model {available[0]}[/dim]"
                    )
            return False

        # 2. Basic reachability via /health (non-fatal if absent)
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            if r.status_code not in (200, 503):
                console.print(f"[yellow]  /health returned {r.status_code}[/yellow]")
        except requests.RequestException:
            pass

        # 3. Try /v1/chat/completions
        chat_payload: dict = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        if self.model:
            chat_payload["model"] = self.model

        # Prefer the chat endpoint for llama.cpp: it applies the model's own chat
        # template and stop tokens. If the probe fails with a model set (e.g. a
        # preset/router build that rejects an unknown model id), retry without it
        # before falling through to the raw /completion path.
        chat_variants = [chat_payload]
        if self.model:
            no_model = {k: v for k, v in chat_payload.items() if k != "model"}
            chat_variants.append(no_model)

        for variant in chat_variants:
            try:
                r = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=variant, timeout=15,
                )
                if r.status_code == 200:
                    self._endpoint_mode = _ENDPOINT_CHAT
                    model_label = f" (model: {self.model})" if self.model else ""
                    console.print(f"[green]  Endpoint:[/green] /v1/chat/completions{model_label} ✓")
                    return True
                else:
                    if self.diagnose:
                        console.print(f"[dim]  /v1/chat/completions → {r.status_code}: {r.text[:400]}[/dim]")
            except requests.RequestException as e:
                if self.diagnose:
                    console.print(f"[dim]  /v1/chat/completions error: {e}[/dim]")

        # 3. Try native /completion endpoint (llama.cpp)
        native_payload: dict = {
            "prompt": "Hi",
            "n_predict": 1,
            "temperature": 0.0,
        }
        if self.model:
            native_payload["model"] = self.model

        try:
            r = self.session.post(
                f"{self.base_url}/completion",
                json=native_payload, timeout=15,
            )
            if r.status_code == 200:
                self._endpoint_mode = _ENDPOINT_NATIVE
                self._server_type = _SERVER_LLAMACPP
                model_label = f" (model: {self.model})" if self.model else ""
                console.print(f"[green]  Endpoint:[/green] /completion (native){model_label} ✓")
                return True
            else:
                if self.diagnose:
                    console.print(f"[dim]  /completion → {r.status_code}: {r.text[:400]}[/dim]")
        except requests.RequestException as e:
            if self.diagnose:
                console.print(f"[dim]  /completion error: {e}[/dim]")

        # 4. If all failed and no model was given, suggest available models
        if not self.model:
            available = self.list_models()
            if available:
                console.print(
                    f"[yellow]  Hint:[/yellow] Server returned model-required error but no --model was given.\n"
                    f"  Available models: {', '.join(available)}\n"
                    f"  Re-run with e.g. [dim]--model {available[0]}[/dim]"
                )

        return False

    # Legacy alias
    def health_check(self) -> bool:
        return self.probe()

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> tuple[str, TimingMetrics, Optional[str]]:
        """Returns (response_text, metrics, error_or_None). Uses detected endpoint."""
        if self._endpoint_mode == _ENDPOINT_NATIVE:
            text, metrics, error = self._complete_native(prompt, system_prompt, max_tokens, temperature)
        elif self._endpoint_mode == _ENDPOINT_OLLAMA:
            text, metrics, error = self._complete_ollama(prompt, system_prompt, max_tokens, temperature)
        else:
            text, metrics, error = self._complete_chat(prompt, system_prompt, max_tokens, temperature)
        # Reasoning models emit chain-of-thought we don't want to score.
        return strip_thinking(text), metrics, error

    def _complete_chat(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, TimingMetrics, Optional[str]]:
        payload: dict = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model:
            payload["model"] = self.model

        wall_start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload, timeout=self.timeout,
            )
            if resp.status_code != 200:
                wall_ms = (time.perf_counter() - wall_start) * 1000
                err = f"HTTP {resp.status_code}"
                if self.diagnose:
                    err += f": {resp.text[:400]}"
                return "", TimingMetrics(total_wall_time_ms=wall_ms), err
        except requests.RequestException as e:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            return "", TimingMetrics(total_wall_time_ms=wall_ms), str(e)

        wall_ms = (time.perf_counter() - wall_start) * 1000
        data = resp.json()
        text = ""
        parse_error = None
        try:
            content = data["choices"][0]["message"]["content"]
            text = content if content is not None else ""
            if content is None:
                parse_error = "response content was null"
                if self.diagnose:
                    console.print(f"[dim]  _complete_chat null content: {str(data)[:400]}[/dim]")
        except (KeyError, IndexError, TypeError) as e:
            parse_error = f"could not extract content: {e}"
            if self.diagnose:
                console.print(f"[dim]  _complete_chat parse error ({e}): {str(data)[:400]}[/dim]")
        return text, self._extract_metrics(data, wall_ms), parse_error

    def _complete_native(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, TimingMetrics, Optional[str]]:
        full_prompt = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        payload: dict = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": _NATIVE_STOP,
            "cache_prompt": False,
        }
        if self.model:
            payload["model"] = self.model

        wall_start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/completion",
                json=payload, timeout=self.timeout,
            )
            if resp.status_code != 200:
                wall_ms = (time.perf_counter() - wall_start) * 1000
                err = f"HTTP {resp.status_code}"
                if self.diagnose:
                    err += f": {resp.text[:400]}"
                return "", TimingMetrics(total_wall_time_ms=wall_ms), err
        except requests.RequestException as e:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            return "", TimingMetrics(total_wall_time_ms=wall_ms), str(e)

        wall_ms = (time.perf_counter() - wall_start) * 1000
        data = resp.json()
        content = data.get("content")
        text = content if content is not None else ""
        parse_error = None
        if content is None:
            parse_error = "response content was null"
            if self.diagnose:
                console.print(f"[dim]  _complete_native null content: {str(data)[:400]}[/dim]")
        return text, self._extract_metrics(data, wall_ms), parse_error

    def _complete_ollama(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, TimingMetrics, Optional[str]]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        wall_start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload, timeout=self.timeout,
            )
            if resp.status_code != 200:
                wall_ms = (time.perf_counter() - wall_start) * 1000
                err = f"HTTP {resp.status_code}"
                if self.diagnose:
                    err += f": {resp.text[:400]}"
                return "", TimingMetrics(total_wall_time_ms=wall_ms), err
        except requests.RequestException as e:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            return "", TimingMetrics(total_wall_time_ms=wall_ms), str(e)

        wall_ms = (time.perf_counter() - wall_start) * 1000
        data = resp.json()
        text = ""
        parse_error = None
        try:
            message = data["message"]
            content = message.get("content")
            # Ignore the separate `thinking` field — we only score the final answer.
            if content:
                text = content
            else:
                parse_error = "response content was null"
                if self.diagnose:
                    console.print(f"[dim]  _complete_ollama null content: {str(data)[:400]}[/dim]")
        except (KeyError, TypeError) as e:
            parse_error = f"could not extract content: {e}"
            if self.diagnose:
                console.print(f"[dim]  _complete_ollama parse error ({e}): {str(data)[:400]}[/dim]")
        return text, self._extract_metrics(data, wall_ms), parse_error

    def _extract_metrics(self, data: dict, wall_ms: float) -> TimingMetrics:
        metrics = TimingMetrics(total_wall_time_ms=wall_ms)

        # llama.cpp native / chat: "timings" dict with millisecond values
        timings = data.get("timings") or {}
        if timings:
            pt    = timings.get("prompt_n",        timings.get("prompt_eval_n", 0))
            pt_ms = timings.get("prompt_ms",        timings.get("prompt_eval_ms", 0.0))
            gt    = timings.get("predicted_n",      timings.get("eval_n", 0))
            gt_ms = timings.get("predicted_ms",     timings.get("eval_ms", 0.0))
            metrics.prompt_tokens       = int(pt)
            metrics.prompt_eval_time_ms = float(pt_ms)
            metrics.prompt_speed_tok_s  = float(timings.get(
                "prompt_per_second", pt / (pt_ms / 1000) if pt_ms > 0 else 0))
            metrics.gen_tokens    = int(gt)
            metrics.gen_time_ms   = float(gt_ms)
            metrics.gen_speed_tok_s = float(timings.get(
                "predicted_per_second", gt / (gt_ms / 1000) if gt_ms > 0 else 0))
            return metrics

        # Ollama /api/chat response: nanosecond duration fields
        # Keys: prompt_eval_count, prompt_eval_duration (ns),
        #       eval_count, eval_duration (ns)
        if "eval_count" in data or "prompt_eval_count" in data:
            pt    = data.get("prompt_eval_count", 0)
            pt_ns = data.get("prompt_eval_duration", 0)
            gt    = data.get("eval_count", 0)
            gt_ns = data.get("eval_duration", 0)
            pt_ms = pt_ns / 1_000_000
            gt_ms = gt_ns / 1_000_000
            metrics.prompt_tokens       = int(pt)
            metrics.prompt_eval_time_ms = float(pt_ms)
            metrics.prompt_speed_tok_s  = pt / (pt_ms / 1000) if pt_ms > 0 else 0.0
            metrics.gen_tokens          = int(gt)
            metrics.gen_time_ms         = float(gt_ms)
            metrics.gen_speed_tok_s     = gt / (gt_ms / 1000) if gt_ms > 0 else 0.0
            return metrics

        # OpenAI-compat fallback: usage only, no speed info
        usage = data.get("usage", {})
        metrics.prompt_tokens = usage.get("prompt_tokens", 0)
        metrics.gen_tokens    = usage.get("completion_tokens", 0)
        return metrics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmarks(
    client: LlamaCppClient,
    exercises: list[Exercise],
    repeat: int,
    verbose: bool,
    categories: Optional[list[str]],
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    filtered = exercises
    if categories:
        cats_lower = [c.lower() for c in categories]
        filtered = [e for e in exercises if e.category.lower() in cats_lower]

    total = len(filtered) * repeat
    done = 0

    for ex in filtered:
        run_metrics: list[TimingMetrics] = []
        run_passed: list[bool] = []
        run_texts: list[str] = []
        run_errors: list[Optional[str]] = []

        for run_i in range(repeat):
            done += 1
            console.print(
                f"[dim]  [{done}/{total}] Exercise {ex.id}: {ex.description} "
                f"(run {run_i + 1}/{repeat})[/dim]"
            )

            text, metrics, error = client.complete(
                prompt=ex.prompt,
                system_prompt=ex.system_prompt,
                max_tokens=ex.max_tokens,
            )

            if error and not text:
                passed = False
            else:
                try:
                    passed = ex.validator(text)
                except Exception as ve:
                    passed = False
                    error = f"Validator error: {ve}"

            run_metrics.append(metrics)
            run_passed.append(passed)
            run_texts.append(text)
            run_errors.append(error)

            if verbose:
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(f"    Response: [italic]{text[:120]}[/italic]")
                console.print(f"    Result: {status}")

        # Majority vote across runs: pass only if a strict majority passed.
        # (Even split counts as a fail, since it isn't a majority.)
        n_pass = sum(run_passed)
        ex_passed = n_pass * 2 > repeat

        # Show a representative response/error that agrees with the verdict:
        # the last passing run if we passed, else the last failing run.
        rep_idx = next(
            (i for i in range(repeat - 1, -1, -1) if run_passed[i] == ex_passed),
            repeat - 1,
        )
        rep_text = run_texts[rep_idx]
        rep_error = run_errors[rep_idx]
        if repeat > 1:
            vote = f"{n_pass}/{repeat} runs passed"
            rep_error = f"{rep_error}; {vote}" if rep_error else (None if ex_passed else vote)

        # Use median metrics across runs
        def median_field(field_name: str) -> float:
            vals = sorted(getattr(m, field_name) for m in run_metrics)
            n = len(vals)
            return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

        median_metrics = TimingMetrics(
            prompt_tokens=run_metrics[-1].prompt_tokens,
            prompt_eval_time_ms=median_field("prompt_eval_time_ms"),
            prompt_speed_tok_s=median_field("prompt_speed_tok_s"),
            gen_tokens=run_metrics[-1].gen_tokens,
            gen_time_ms=median_field("gen_time_ms"),
            gen_speed_tok_s=median_field("gen_speed_tok_s"),
            total_wall_time_ms=median_field("total_wall_time_ms"),
        )

        results.append(BenchmarkResult(
            exercise_id=ex.id,
            category=ex.category,
            description=ex.description,
            passed=ex_passed,
            response_text=rep_text,
            metrics=median_metrics,
            error=rep_error,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Math":        "cyan",
    "Factual":     "blue",
    "Reasoning":   "magenta",
    "Instruction": "yellow",
    "Code":        "green",
    "Extraction":  "red",
}


def print_report(results: list[BenchmarkResult], model_name: str):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = (passed / total * 100) if total else 0

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        title=f"[bold]LLM Benchmark — {model_name} — {datetime.now().strftime('%Y-%m-%d %H:%M')}[/bold]",
        expand=True,
    )
    table.add_column("#",       style="dim", width=4, justify="right")
    table.add_column("Category", width=12)
    table.add_column("Description", min_width=30)
    table.add_column("Result",  width=6, justify="center")
    table.add_column("PP tok/s", width=10, justify="right")
    table.add_column("TG tok/s", width=10, justify="right")
    table.add_column("Wall ms",  width=9, justify="right")
    table.add_column("Error", style="dim red", min_width=20)

    for r in results:
        color = CATEGORY_COLORS.get(r.category, "white")
        result_text = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        pp = f"{r.metrics.prompt_speed_tok_s:,.0f}" if r.metrics.prompt_speed_tok_s else "—"
        tg = f"{r.metrics.gen_speed_tok_s:,.0f}" if r.metrics.gen_speed_tok_s else "—"
        wall = f"{r.metrics.total_wall_time_ms:,.0f}"
        err = (r.error or "")[:40]

        table.add_row(
            str(r.exercise_id),
            f"[{color}]{r.category}[/{color}]",
            r.description,
            result_text,
            pp,
            tg,
            wall,
            err,
        )

    console.print()
    console.print(table)

    # Aggregate stats
    pp_vals = [r.metrics.prompt_speed_tok_s for r in results if r.metrics.prompt_speed_tok_s > 0]
    tg_vals = [r.metrics.gen_speed_tok_s for r in results if r.metrics.gen_speed_tok_s > 0]
    wall_vals = [r.metrics.total_wall_time_ms for r in results]

    def median(lst):
        if not lst: return 0.0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    score_color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
    summary = (
        f"[bold {score_color}]SCORE: {passed}/{total} ({pct:.0f}%)[/bold {score_color}]"
        f"   Median PP: [cyan]{median(pp_vals):,.0f} tok/s[/cyan]"
        f"   Median TG: [cyan]{median(tg_vals):,.0f} tok/s[/cyan]"
        f"   Median Wall: [cyan]{median(wall_vals):,.0f} ms[/cyan]"
    )
    console.print(Panel(summary, expand=False))

    # Per-category breakdown
    categories = sorted(set(r.category for r in results))
    cat_table = Table(box=box.SIMPLE, title="[bold]Per-Category Breakdown[/bold]", expand=False)
    cat_table.add_column("Category")
    cat_table.add_column("Pass", justify="center")
    cat_table.add_column("Total", justify="center")
    cat_table.add_column("%", justify="right")

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cp = sum(1 for r in cat_results if r.passed)
        ct = len(cat_results)
        cpct = cp / ct * 100 if ct else 0
        color = CATEGORY_COLORS.get(cat, "white")
        pct_color = "green" if cpct >= 80 else "yellow" if cpct >= 60 else "red"
        cat_table.add_row(
            f"[{color}]{cat}[/{color}]",
            str(cp), str(ct),
            f"[{pct_color}]{cpct:.0f}%[/{pct_color}]"
        )

    console.print()
    console.print(cat_table)
    console.print()


def save_results(results: list[BenchmarkResult], path: str, model_name: str):
    output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "score": {
            "passed": sum(1 for r in results if r.passed),
            "total": len(results),
        },
        "results": [
            {
                **{k: v for k, v in asdict(r).items() if k != "metrics"},
                "metrics": asdict(r.metrics),
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"[dim]Results saved to {path}[/dim]")


# ---------------------------------------------------------------------------
# WSL2 networking helpers
# ---------------------------------------------------------------------------
# A llama.cpp server bound to 0.0.0.0 on the *Windows* host is not reachable
# from inside WSL2 via localhost/127.0.0.1 — those resolve to the WSL VM, not
# the host. The Windows host is reachable at the default-route gateway IP.

def _is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _wsl_windows_host() -> Optional[str]:
    """Windows host IP as seen from WSL2 (the default-route gateway)."""
    if not _is_wsl():
        return None
    try:
        out = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        m = re.search(r"via (\d+\.\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a llama.cpp or Ollama server — measures PP/TG speeds and correctness."
    )
    parser.add_argument("--host",       default="localhost",  help="Server host (default: localhost)")
    parser.add_argument("--port",       default=None, type=int, help="Server port (default: 8080 for llama.cpp, 11434 for Ollama)")
    parser.add_argument("--model",      default="local",      help="Model name label for output (default: local)")
    parser.add_argument("--ollama",     action="store_true",  help="Use the Ollama backend (/api/chat); otherwise the llama.cpp endpoints are used")
    parser.add_argument("--repeat",     default=1,  type=int, help="Runs per exercise — median is reported (default: 1)")
    parser.add_argument("--categories", default=None,         help="Comma-separated categories to run (e.g. Math,Code)")
    parser.add_argument("--output",     default=None,         help="Save JSON results to this file")
    parser.add_argument("--verbose",    action="store_true",  help="Print each model response")
    parser.add_argument("--diagnose",   action="store_true",  help="Print raw server error bodies to debug 400s")
    parser.add_argument("--list-models", action="store_true",  dest="list_models",
                        help="Query the server for available model names and exit")
    parser.add_argument("--list",       action="store_true",  help="List all exercises and exit")
    args = parser.parse_args()

    # Port defaults depend on the server type: 11434 for Ollama, 8080 for llama.cpp.
    if args.port is None:
        args.port = 11434 if args.ollama else 8080

    exercises = build_exercises()

    if args.list:
        for ex in exercises:
            console.print(f"  [{ex.id:02d}] [{ex.category}] {ex.description}")
        return

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    console.print("\n[bold]LLM Benchmark (llama.cpp / Ollama)[/bold]")
    console.print(f"  Server : [cyan]{args.host}:{args.port}[/cyan]")
    console.print(f"  Backend: [cyan]{'Ollama' if args.ollama else 'llama.cpp'}[/cyan]")
    console.print(f"  Model  : [cyan]{args.model}[/cyan]")
    console.print(f"  Repeats: [cyan]{args.repeat}[/cyan]")
    if categories:
        console.print(f"  Filter : [cyan]{', '.join(categories)}[/cyan]")
    console.print()

    client = LlamaCppClient(
        host=args.host,
        port=args.port,
        model=args.model,
        diagnose=args.diagnose,
        ollama=args.ollama,
    )

    # --list-models: just print available models and exit
    if args.list_models:
        models = client.list_models()
        if models:
            console.print("\n[bold]Available models on this server:[/bold]")
            for m in models:
                console.print(f"  • {m}")
            console.print("\nUse [dim]--model <name>[/dim] to target one.")
        else:
            console.print("[yellow]No models returned by /v1/models (server may not support listing).[/yellow]")
        return

    console.print("[dim]Probing server — detecting endpoint...[/dim]")
    ok = client.probe()
    if not ok:
        # WSL2: localhost won't reach a server bound on the Windows host. If the
        # given host isn't even accepting TCP connections but the Windows gateway
        # is, retry there before giving up.
        win_host = _wsl_windows_host()
        if (win_host and win_host != args.host
                and args.host in ("localhost", "127.0.0.1")
                and not _port_open(args.host, args.port)
                and _port_open(win_host, args.port)):
            console.print(
                f"[yellow]  {args.host}:{args.port} unreachable — detected WSL2; "
                f"retrying on Windows host [cyan]{win_host}[/cyan][/yellow]"
            )
            args.host = win_host
            client = LlamaCppClient(
                host=args.host,
                port=args.port,
                model=args.model,
                diagnose=args.diagnose,
                ollama=args.ollama,
            )
            ok = client.probe()

    if not ok:
        available = client.list_models()
        hint = ""
        if available:
            hint = (
                f"\n  Available models: [cyan]{', '.join(available)}[/cyan]"
                f"\n  Re-run with e.g. [dim]--model {available[0]}[/dim]"
            )
        else:
            hint = (
                "\n  Run [dim]--list-models[/dim] to see what model names the server expects."
                "\n  Add [dim]--diagnose[/dim] to print the raw server error body."
            )
        console.print(
            f"\n[bold red]ERROR:[/bold red] Could not connect to a working endpoint at "
            f"[cyan]{args.host}:{args.port}[/cyan].\n"
            f"  Tried /v1/chat/completions, /completion, and /api/chat — all failed.{hint}"
        )
        sys.exit(1)
    console.print("[green]Server is up.[/green]\n")

    console.print("[bold]Running exercises...[/bold]")
    results = run_benchmarks(client, exercises, args.repeat, args.verbose, categories)

    print_report(results, args.model)

    if args.output:
        save_results(results, args.output, args.model)


if __name__ == "__main__":
    main()