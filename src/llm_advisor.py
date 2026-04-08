"""
LLM-assisted query suggestions for BBO capstone (Week 8+).

Provider-agnostic client (OpenAI or Anthropic), prompt builders, response parsing,
and experiment logging for scientific comparison of prompts and decoding settings.
"""
from __future__ import annotations

import ast
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .data import DATA_DIR, FunctionData

# Domain blurbs distilled from notes/BBO_Strategy_W7_Report.md (Sections 3–10).
DOMAIN_CONTEXTS: Dict[int, str] = {
    1: (
        "Domain: 2D radiation / contamination detection. Signal is extremely localised "
        "(inverse-square style); most reads are near zero. Goal: maximise reading."
    ),
    2: (
        "Domain: noisy ML log-likelihood (2D). Multiple local optima and ridge-like trade-offs "
        "between parameters are plausible."
    ),
    3: (
        "Domain: drug discovery / compound dosing (3D). Therapeutic windows and antagonistic "
        "interactions between compounds are plausible; avoid extreme doses."
    ),
    4: (
        "Domain: warehouse ML hyperparameters (4D). Sharp cliffs and a narrow peak are expected; "
        "small moves can collapse performance."
    ),
    5: (
        "Domain: chemical process yield (4D). Described as unimodal with peak near upper boundary; "
        "all inputs in [0,1] typically correlate positively with yield."
    ),
    6: (
        "Domain: cake recipe (5D: flour, sugar, eggs, butter, milk). Liquid (milk) and sugar×liquid "
        "interactions are often harmful when high; butter/eggs can synergise."
    ),
    7: (
        "Domain: generic ML model hyperparameters (6D). Low learning rate, moderate regularisation, "
        "low dropout, and sufficient capacity often work well; high LR is risky."
    ),
    8: (
        "Domain: 8D neural net tuning — x0 LR, x1 batch, x2 layers, x3 dropout, x4 regularisation, "
        "x5 activation code, x6 optimiser code, x7 init scale. Shallow + well-regularised configs "
        "often win; high LR × many layers × bad optimiser is dangerous."
    ),
}

JSON_INSTRUCTION = (
    "Respond with a single JSON object only, no markdown fences, with exactly these keys:\n"
    '  "query": [list of D numbers each in [0, 1]],\n'
    '  "reasoning": "brief string"\n'
    "Use at most 4 decimal places for numbers in the query list."
)


def _format_scalar_for_prompt(v: float) -> str:
    """Avoid scientific notation in prompts (tokenisation); keep readable precision."""
    if not np.isfinite(v):
        return "0.0000"
    av = abs(float(v))
    if av != 0 and av < 1e-4:
        return f"{float(v):.8f}".rstrip("0").rstrip(".") or "0.0000"
    return f"{float(v):.4f}"


@dataclass
class LLMResponse:
    """Normalised completion metadata from any provider."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    """
    Thin wrapper for OpenAI Chat Completions or Anthropic Messages API.

    Environment variables:
      OPENAI_API_KEY  — when provider is \"openai\"
      ANTHROPIC_API_KEY — when provider is \"anthropic\"
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
    ) -> None:
        p = provider.lower().strip()
        if p not in ("openai", "anthropic"):
            raise ValueError('provider must be \"openai\" or \"anthropic\"')
        self.provider = p
        if p == "openai":
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
        else:
            self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        max_tokens: int = 500,
    ) -> LLMResponse:
        if self.provider == "openai":
            return self._complete_openai(prompt, temperature, top_p, max_tokens)
        return self._complete_anthropic(prompt, temperature, top_p, top_k, max_tokens)

    def _complete_openai(
        self, prompt: str, temperature: float, top_p: float, max_tokens: int
    ) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai package: pip install openai>=1.0.0") from e

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = getattr(resp, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        return LLMResponse(text=text.strip(), model=self.model, prompt_tokens=pt, completion_tokens=ct)

    def _complete_anthropic(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        max_tokens: int,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Install anthropic package: pip install anthropic>=0.18.0") from e

        client = anthropic.Anthropic()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if top_k is not None:
            kwargs["top_k"] = top_k
        msg = client.messages.create(**kwargs)
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        text = "".join(parts).strip()
        pt = getattr(msg.usage, "input_tokens", 0) or 0
        ct = getattr(msg.usage, "output_tokens", 0) or 0
        return LLMResponse(text=text, model=self.model, prompt_tokens=int(pt), completion_tokens=int(ct))


class PromptBuilder:
    """Build prompts from FunctionData; optional domain context and data subsetting."""

    def __init__(self, include_json_instruction: bool = True) -> None:
        self.include_json_instruction = include_json_instruction

    def _footer(self) -> str:
        return JSON_INSTRUCTION if self.include_json_instruction else ""

    def _domain_block(self, func_id: int, use_domain: bool) -> str:
        if not use_domain:
            return ""
        ctx = DOMAIN_CONTEXTS.get(func_id, "")
        if not ctx:
            return ""
        return f"\nContext (may help interpret patterns; verify against data):\n{ctx}\n"

    def _data_table(
        self, func_data: FunctionData, max_points: Optional[int]
    ) -> Tuple[str, int]:
        X, y = func_data.inputs, func_data.outputs
        n = len(y)
        if max_points is not None and max_points < n:
            # Keep best point + most recent points to study recency bias in the LLM.
            best_i = int(np.argmax(y))
            idx_rest = [i for i in range(n) if i != best_i]
            tail = idx_rest[-(max_points - 1) :] if max_points > 1 else []
            idx = [best_i] + tail
            idx = sorted(set(idx), key=lambda i: i)
            X, y = X[idx], y[idx]
        header = "i\t" + "\t".join(f"x{j}" for j in range(func_data.n_dims)) + "\ty"
        rows = [header]
        for i in range(len(y)):
            xs = "\t".join(_format_scalar_for_prompt(float(X[i, j])) for j in range(func_data.n_dims))
            rows.append(f"{i}\t{xs}\t{_format_scalar_for_prompt(float(y[i]))}")
        return "\n".join(rows), len(y)

    def zero_shot(self, func_data: FunctionData, use_domain: bool = False) -> str:
        fid = func_data.function_id
        d = func_data.n_dims
        best_x, best_y = func_data.get_best()
        bx = ", ".join(_format_scalar_for_prompt(float(best_x[j])) for j in range(d))
        body = (
            f"You are helping choose the next query for a black-box function to MAXIMISE its output.\n"
            f"Function id: {fid}, dimension D={d}. Each coordinate must be in [0, 1].\n"
            f"Current best observed point (coordinates → value): [{bx}] → {_format_scalar_for_prompt(float(best_y))}.\n"
            f"Suggest one new query (vector of length {d}) that is most likely to improve the best value.\n"
        )
        body += self._domain_block(fid, use_domain)
        body += "\n" + self._footer()
        return body.strip()

    def few_shot(
        self,
        func_data: FunctionData,
        use_domain: bool = False,
        max_points: Optional[int] = None,
    ) -> str:
        fid = func_data.function_id
        d = func_data.n_dims
        table, n_used = self._data_table(func_data, max_points)
        body = (
            f"You are helping choose the next query for a black-box function to MAXIMISE its output.\n"
            f"Function id: {fid}, dimension D={d}. Each coordinate must be in [0, 1].\n"
            f"Below are {n_used} past observations (tab-separated). Higher y is better.\n\n"
            f"{table}\n\n"
            f"Suggest one new query (vector of length {d}) not identical to any row.\n"
        )
        body += self._domain_block(fid, use_domain)
        body += "\n" + self._footer()
        return body.strip()

    def chain_of_thought(
        self,
        func_data: FunctionData,
        use_domain: bool = False,
        max_points: Optional[int] = None,
    ) -> str:
        base = self.few_shot(func_data, use_domain=use_domain, max_points=max_points)
        insert = (
            "\nFirst, write a short numbered list of reasoning steps in plain text.\n"
            "Then output the JSON object on the last line exactly as specified.\n"
        )
        return (base + insert).strip()


class ResponseParser:
    """Parse model output into a bounded query vector."""

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def parse(self, text: str, expected_dim: int) -> Tuple[Optional[np.ndarray], bool, str]:
        """
        Returns (query_array_or_None, success, message).
        """
        raw = text.strip()
        # Strip optional markdown ```json ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if fence:
            raw = fence.group(1).strip()

        obj: Optional[Dict[str, Any]] = None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find first JSON object in text
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = None

        if obj is None:
            arr = self._try_python_list(raw) or self._try_csv(raw, expected_dim)
            if arr is not None:
                return self._validate(arr, expected_dim)
            return None, False, "Could not parse JSON or fallback list from model output."

        q = obj.get("query")
        if q is None:
            return None, False, 'JSON missing "query" key.'
        try:
            arr = np.array([float(q[i]) for i in range(len(q))], dtype=float)
        except (TypeError, ValueError, IndexError):
            return None, False, '"query" is not a list of numbers.'

        return self._validate(arr, expected_dim)

    def _try_python_list(self, text: str) -> Optional[np.ndarray]:
        m = re.search(r"\[[\s0-9.,+\-eE]+\]", text)
        if not m:
            return None
        try:
            val = ast.literal_eval(m.group(0))
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=float)
        except (ValueError, SyntaxError):
            return None
        return None

    def _try_csv(self, text: str, d: int) -> Optional[np.ndarray]:
        # Last line: comma-separated floats
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,;]+", line)
            try:
                nums = [float(p) for p in parts if p]
            except ValueError:
                continue
            if len(nums) == d:
                return np.array(nums, dtype=float)
        return None

    def _validate(self, arr: np.ndarray, expected_dim: int) -> Tuple[Optional[np.ndarray], bool, str]:
        if arr.ndim != 1 or arr.shape[0] != expected_dim:
            return (
                None,
                False,
                f"Expected length {expected_dim}, got {arr.shape[0] if arr.ndim == 1 else arr.shape}.",
            )
        if np.any(~np.isfinite(arr)):
            return None, False, "Non-finite values in query."
        if np.any(arr < -self.tol) or np.any(arr > 1.0 + self.tol):
            return None, False, "Query outside [0, 1] bounds."
        clipped = np.clip(arr, 0.0, 0.999999)
        return clipped, True, "ok"


class ExperimentLogger:
    """Append-only JSON log of LLM experiment runs."""

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        base = DATA_DIR / "results" / "week_8"
        base.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path is not None else base / "llm_experiments.json"
        self.records: List[Dict[str, Any]] = []
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.records = data
            except (json.JSONDecodeError, OSError):
                self.records = []

    def append(self, record: Dict[str, Any]) -> None:
        record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.records.append(record)
        self._flush()

    def _flush(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def to_dataframe(self):
        """Return a pandas DataFrame if pandas is installed; else the raw list of dicts."""
        try:
            import pandas as pd

            return pd.DataFrame(self.records)
        except ImportError:
            warnings.warn("pandas not installed; returning list of records")
            return self.records


def surrogate_mean_at(
    func_data: FunctionData,
    x: np.ndarray,
    surrogate_kind: str = "gp",
    use_ard: bool = True,
) -> Optional[float]:
    """
    GP or SVR mean prediction at x (for logging agreement with LLM suggestions).
    """
    from .surrogates import GPSurrogate, SVMSurrogate

    X, y = func_data.inputs, func_data.outputs
    if len(y) < 2:
        return None
    x = np.asarray(x, dtype=float).reshape(1, -1)
    try:
        if surrogate_kind == "svr":
            model = SVMSurrogate()
        else:
            model = GPSurrogate(use_ard=use_ard, optimize=True)
        model.fit(X, y)
        mean, _ = model.predict(x)
        return float(mean[0])
    except Exception as e:
        warnings.warn(f"surrogate_mean_at failed: {e}")
        return None
