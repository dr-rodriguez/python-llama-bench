#!/usr/bin/env python3
"""
llama_bench.py — Benchmark script for llama.cpp servers
Measures prompt-processing speed, token generation speed, TTFT, and wall time
across 25 exercises with verifiable results.
"""

import argparse
import json
import re
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
            description="Capital of Japan",
            prompt="What is the capital of Japan? Reply with only the city name.",
            validator=exact_match("Tokyo"),
        ),
        Exercise(
            id=8, category="Factual",
            description="Chemical symbol for gold",
            prompt="What is the chemical symbol for gold? Reply with only the symbol.",
            validator=exact_match("Au"),
        ),
        Exercise(
            id=9, category="Factual",
            description="Number of planets in the solar system",
            prompt="How many planets are in our solar system? Reply with only the number.",
            validator=starts_with_number(8),
        ),
        Exercise(
            id=10, category="Factual",
            description="Boiling point of water in Celsius",
            prompt="At what temperature in Celsius does water boil at sea level? Reply with only the number.",
            validator=starts_with_number(100),
        ),
        Exercise(
            id=11, category="Factual",
            description="Acronym: CPU",
            prompt="What does the acronym CPU stand for? Reply with only the expanded form.",
            validator=contains_all("central", "processing", "unit"),
        ),

        # ── Reasoning & Language ─────────────────────────────────────────────
        Exercise(
            id=12, category="Reasoning",
            description="Odd one out (carrot among fruits)",
            prompt="Which word does not belong with the others? apple, banana, carrot, grape. Reply with only the word.",
            validator=exact_match("carrot"),
        ),
        Exercise(
            id=13, category="Reasoning",
            description="Analogy: hot:cold :: day:___",
            prompt="Complete the analogy. Hot is to cold as day is to ___. Reply with only the missing word.",
            validator=exact_match("night"),
        ),
        Exercise(
            id=14, category="Reasoning",
            description="Next in sequence: 2, 4, 8, 16, ___",
            prompt="What is the next number in the sequence: 2, 4, 8, 16, ___? Reply with only the number.",
            validator=starts_with_number(32),
        ),
        Exercise(
            id=15, category="Reasoning",
            description="Word that rhymes with 'cat'",
            prompt="Give exactly one word that rhymes with 'cat'. Reply with only that word.",
            validator=ends_with_pattern("at"),
        ),
        Exercise(
            id=16, category="Reasoning",
            description="Acronym: DNA",
            prompt="What does DNA stand for? Reply with only the expanded form.",
            validator=contains_all("deoxyribonucleic", "acid"),
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
        ),

        # ── Code Generation ───────────────────────────────────────────────────
        Exercise(
            id=21, category="Code",
            description="Python: squares list comprehension",
            prompt="Write a single Python print statement using a list comprehension that prints the squares of 1 through 5. Provide only the code, no explanation.",
            validator=python_code_produces("[1, 4, 9, 16, 25]"),
        ),
        Exercise(
            id=22, category="Code",
            description="Python: reverse a string",
            prompt='Write a single Python print statement that prints the reverse of the string "hello". Provide only the code, no explanation.',
            validator=python_code_produces("olleh"),
        ),
        Exercise(
            id=23, category="Code",
            description="Python: FizzBuzz output for n=15",
            prompt="Write a Python snippet that prints FizzBuzz for numbers 1 through 15 (print Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both). Provide only the code.",
            validator=python_code_produces("FizzBuzz"),
        ),
        Exercise(
            id=24, category="Code",
            description="Python: sum of a list",
            prompt="Write a single Python print statement that prints the sum of the list [10, 20, 30, 40]. Provide only the code, no explanation.",
            validator=python_code_produces("100"),
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
        ),
    ]


# ---------------------------------------------------------------------------
# llama.cpp HTTP client
# ---------------------------------------------------------------------------

_ENDPOINT_CHAT    = "chat"     # POST /v1/chat/completions  (OpenAI-compat)
_ENDPOINT_NATIVE  = "native"   # POST /completion           (llama.cpp native)


class LlamaCppClient:
    def __init__(
        self,
        host: str,
        port: int,
        model: str = "",
        timeout: int = 120,
        diagnose: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"
        self.model = model          # passed through in every request payload
        self.timeout = timeout
        self.diagnose = diagnose
        self.session = requests.Session()
        self._endpoint_mode: Optional[str] = None   # set by probe()

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Query /v1/models and return the list of available model IDs."""
        try:
            r = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [m.get("id", "") for m in data.get("data", [])]
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
        # 1. Basic reachability via /health (non-fatal if absent)
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            if r.status_code not in (200, 503):
                console.print(f"[yellow]  /health returned {r.status_code}[/yellow]")
        except requests.RequestException:
            pass

        # 2. Try /v1/chat/completions
        chat_payload: dict = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        if self.model:
            chat_payload["model"] = self.model

        try:
            r = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=chat_payload, timeout=15,
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

        # 3. Try native /completion endpoint
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
                model_label = f" (model: {self.model})" if self.model else ""
                console.print(f"[green]  Endpoint:[/green] /completion (native){model_label} ✓")
                return True
            else:
                if self.diagnose:
                    console.print(f"[dim]  /completion → {r.status_code}: {r.text[:400]}[/dim]")
        except requests.RequestException as e:
            if self.diagnose:
                console.print(f"[dim]  /completion error: {e}[/dim]")

        # 4. If both failed and no model was given, suggest available models
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
            return self._complete_native(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._complete_chat(prompt, system_prompt, max_tokens, temperature)

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
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass
        return text, self._extract_metrics(data, wall_ms), None

    def _complete_native(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, TimingMetrics, Optional[str]]:
        full_prompt = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        payload: dict = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
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
        text = data.get("content", "")
        return text, self._extract_metrics(data, wall_ms), None

    def _extract_metrics(self, data: dict, wall_ms: float) -> TimingMetrics:
        metrics = TimingMetrics(total_wall_time_ms=wall_ms)
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
        else:
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
        last_text = ""
        last_error = None
        last_passed = False

        for run_i in range(repeat):
            done += 1
            console.print(
                f"[dim]  [{done}/{total}] Exercise {ex.id}: {ex.description} "
                f"(run {run_i + 1}/{repeat})[/dim]"
            )

            text, metrics, error = client.complete(
                prompt=ex.prompt,
                system_prompt=ex.system_prompt,
            )
            last_text = text
            last_error = error

            if error:
                last_passed = False
            else:
                try:
                    last_passed = ex.validator(text)
                except Exception as ve:
                    last_passed = False
                    last_error = f"Validator error: {ve}"

            run_metrics.append(metrics)

            if verbose:
                status = "[green]PASS[/green]" if last_passed else "[red]FAIL[/red]"
                console.print(f"    Response: [italic]{text[:120]}[/italic]")
                console.print(f"    Result: {status}")

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
            passed=last_passed,
            response_text=last_text,
            metrics=median_metrics,
            error=last_error,
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
        title=f"[bold]llama.cpp Benchmark — {model_name} — {datetime.now().strftime('%Y-%m-%d %H:%M')}[/bold]",
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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a llama.cpp server — measures PP/TG speeds and correctness."
    )
    parser.add_argument("--host",       default="localhost",  help="llama.cpp server host (default: localhost)")
    parser.add_argument("--port",       default=8080, type=int, help="llama.cpp server port (default: 8080)")
    parser.add_argument("--model",      default="local",      help="Model name label for output (default: local)")
    parser.add_argument("--repeat",     default=1,  type=int, help="Runs per exercise — median is reported (default: 1)")
    parser.add_argument("--categories", default=None,         help="Comma-separated categories to run (e.g. Math,Code)")
    parser.add_argument("--output",     default=None,         help="Save JSON results to this file")
    parser.add_argument("--verbose",    action="store_true",  help="Print each model response")
    parser.add_argument("--diagnose",   action="store_true",  help="Print raw server error bodies to debug 400s")
    parser.add_argument("--list-models", action="store_true",  dest="list_models",
                        help="Query the server for available model names and exit")
    parser.add_argument("--list",       action="store_true",  help="List all exercises and exit")
    args = parser.parse_args()

    exercises = build_exercises()

    if args.list:
        for ex in exercises:
            console.print(f"  [{ex.id:02d}] [{ex.category}] {ex.description}")
        return

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    console.print(f"\n[bold]llama.cpp Benchmark[/bold]")
    console.print(f"  Server : [cyan]{args.host}:{args.port}[/cyan]")
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
    )

    # --list-models: just print available models and exit
    if args.list_models:
        models = client.list_models()
        if models:
            console.print("\n[bold]Available models on this server:[/bold]")
            for m in models:
                console.print(f"  • {m}")
            console.print(f"\nUse [dim]--model <name>[/dim] to target one.")
        else:
            console.print("[yellow]No models returned by /v1/models (server may not support listing).[/yellow]")
        return

    console.print("[dim]Probing server — detecting endpoint...[/dim]")
    if not client.probe():
        available = client.list_models()
        hint = ""
        if available:
            hint = (
                f"\n  Available models: [cyan]{', '.join(available)}[/cyan]"
                f"\n  Re-run with e.g. [dim]--model {available[0]}[/dim]"
            )
        else:
            hint = (
                f"\n  Run [dim]--list-models[/dim] to see what model names the server expects."
                f"\n  Add [dim]--diagnose[/dim] to print the raw server error body."
            )
        console.print(
            f"\n[bold red]ERROR:[/bold red] Could not connect to a working endpoint at "
            f"[cyan]{args.host}:{args.port}[/cyan].\n"
            f"  Tried /v1/chat/completions and /completion — both failed.{hint}"
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