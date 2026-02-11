#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

# Optional providers
try:
    import ollama as ollama_pkg  # type: ignore
    OLLAMA_IMPORT_ERROR = ""
except Exception:
    ollama_pkg = None  # type: ignore
    OLLAMA_IMPORT_ERROR = str(sys.exc_info()[1] or "")

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
    OPENAI_IMPORT_ERROR = ""
except Exception:
    OpenAI = None  # type: ignore
    OPENAI_IMPORT_ERROR = str(sys.exc_info()[1] or "")

load_dotenv()

# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────
def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


CONVENTIONAL_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "build",
    "ci",
    "perf",
    "revert",
)

CONVENTIONAL_COMMITS_PATTERN = re.compile(
    rf'^({"|".join(CONVENTIONAL_TYPES)})(\([\w./-]+\))?!?: .+$'
)

CONVENTIONAL_HEADER_PATTERN = re.compile(
    rf'^({"|".join(CONVENTIONAL_TYPES)})(\([\w./-]+\))?(!)?: (.+)$',
    re.IGNORECASE,
)

EMBEDDED_CONVENTIONAL_PATTERN = re.compile(
    rf'({"|".join(CONVENTIONAL_TYPES)})(\([\w./-]+\))?!?: .+$',
    re.IGNORECASE,
)

MESSAGE_PREFIX_PATTERN = re.compile(
    r"^(commit message|message|commit)\s*[:\-]\s*",
    re.IGNORECASE,
)

SUPPORTED_PROVIDERS = ("ollama", "gemini", "openai")

# Ollama (default)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "gemma3n:e4b")

# Gemini
GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-0125")

MAX_DIFF_CHARS = _read_int_env("AICOMMIT_MAX_DIFF_CHARS", 14000)

# ───────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ───────────────────────────────────────────────────────────────────────────────
def _log(msg: str, debug: bool):
    if debug:
        sys.stderr.write(f"[commit-gen] {msg}\n")
        sys.stderr.flush()


def _error(msg: str) -> None:
    sys.stderr.write(f"[commit-gen] {msg}\n")
    sys.stderr.flush()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Conventional Commit message from staged Git changes."
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        help="Force one provider instead of trying the default fallback chain.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language for the generated commit subject (default: en).",
    )
    parser.add_argument(
        "--allow-nonconventional",
        action="store_true",
        help="Allow responses that do not match Conventional Commits.",
    )
    parser.add_argument(
        "--diff-per-file",
        action="store_true",
        help="Build prompt by concatenating per-file diffs.",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=MAX_DIFF_CHARS,
        help=(
            "Maximum staged diff size sent to the model. "
            "Use 0 to disable truncation."
        ),
    )
    parser.add_argument(
        "--no-auto-stage",
        dest="auto_stage",
        action="store_false",
        help="Do not run 'git add -u' before reading the staged diff.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs to stderr.",
    )
    parser.set_defaults(auto_stage=True)
    return parser.parse_args(argv)


# ───────────────────────────────────────────────────────────────────────────────
# Providers (always try AI; no non-AI fallback)
# ───────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Provider:
    name: str  # "ollama" | "gemini" | "openai"
    model: str
    api_key: str = ""
    host: str = ""  # for ollama

def _normalize_ollama_host(host: str) -> str:
    value = (host or "").strip().rstrip("/")
    if value.lower().endswith("/v1"):
        value = value[:-3]
    return value or "http://127.0.0.1:11434"


def _extract_ollama_text(resp: Any) -> str:
    # SDK returns pydantic models; older versions may return dicts.
    if resp is None:
        return ""
    if isinstance(resp, dict):
        return ((resp.get("message") or {}).get("content") or "").strip()
    msg = getattr(resp, "message", None)
    return (getattr(msg, "content", "") or "").strip()


def _list_ollama_models(client: Any) -> List[str]:
    try:
        result = client.list()
    except Exception:
        return []
    models = getattr(result, "models", None)
    if models is None and isinstance(result, dict):
        models = result.get("models")
    out: List[str] = []
    for item in models or []:
        if isinstance(item, dict):
            name = item.get("model") or item.get("name")
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _ollama_model_rank(name: str) -> Tuple[int, int]:
    low = name.lower()
    is_cloud = 1 if low.endswith(":cloud") else 0
    preferred_bases = (
        "gpt-oss",
        "qwen",
        "mistral",
        "llama",
        "gemma",
        "phi",
        "deepseek",
        "codellama",
    )
    pref_index = len(preferred_bases)
    for idx, base in enumerate(preferred_bases):
        if low.startswith(base):
            pref_index = idx
            break
    return (is_cloud, pref_index)


def _build_ollama_candidates(requested: str, installed: List[str]) -> List[str]:
    if not installed:
        return [requested] if requested else []

    ordered: List[str] = []

    def add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    req = (requested or "").strip()
    add(req)

    # Same base tags first (if the requested base exists locally).
    req_base = req.split(":", 1)[0] if req else ""
    if req_base:
        for name in installed:
            if name.split(":", 1)[0] == req_base:
                add(name)

    # Then fall back to best locally installed models.
    indexed = list(enumerate(installed))
    indexed.sort(key=lambda pair: (_ollama_model_rank(pair[1]), pair[0]))
    for _, name in indexed:
        add(name)

    return ordered


def _ollama_chat(model: str, host: str, system_prompt: str, user_prompt: str, debug: bool) -> Tuple[bool, str, str]:
    if ollama_pkg is None:
        detail = OLLAMA_IMPORT_ERROR or "import failed"
        return False, "", f"ollama sdk missing: {detail}"

    host = _normalize_ollama_host(host)
    client = ollama_pkg.Client(host=host)
    options = {"temperature": 0.2, "num_predict": 128}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    installed = _list_ollama_models(client)
    candidates = _build_ollama_candidates(model, installed)
    if not candidates:
        candidates = [model]

    errors: List[str] = []
    response_error = getattr(ollama_pkg, "ResponseError", None)
    for index, candidate in enumerate(candidates):
        if index > 0:
            _log(f"ollama retry with model '{candidate}'", debug)
        try:
            resp = client.chat(
                model=candidate,
                messages=messages,
                options=options,
            )
            text = _extract_ollama_text(resp)
            if text:
                if candidate != model:
                    _log(
                        f"ollama model '{model}' unavailable/empty; using '{candidate}'",
                        debug,
                    )
                return True, text, ""
            errors.append(f"{candidate}: empty")
        except Exception as e:
            if response_error is not None and isinstance(e, response_error):
                status = getattr(e, "status_code", None)
                if status == 404:
                    errors.append(f"{candidate}: not found")
                    continue
                errors.append(f"{candidate}: {e}")
                continue
            errors.append(f"{candidate}: {e}")

    hint = ", ".join(installed[:5]) if installed else "no installed models found"
    summary = "; ".join(errors[:4]) if errors else "no successful response"
    return False, "", f"ollama failed (host={host}): {summary}. installed: {hint}"

def _gemini_complete(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Tuple[bool, str, str]:
    try:
        if genai is None:
            return False, "", "gemini sdk missing"
        genai.configure(api_key=api_key)  # type: ignore
        m = genai.GenerativeModel(model_name=model)  # type: ignore
        r = m.generate_content([system_prompt, user_prompt])  # type: ignore
        return True, (getattr(r, "text", "") or "").strip(), ""
    except Exception as e:
        return False, "", f"gemini error: {e}"

def _openai_complete(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Tuple[bool, str, str]:
    try:
        if OpenAI is None:
            detail = OPENAI_IMPORT_ERROR or "import failed"
            return False, "", f"openai sdk missing: {detail}"
        client = OpenAI(api_key=api_key)  # type: ignore
        out = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=128,
            top_p=1,
        )
        msg = out.choices[0].message.content if out.choices else ""
        return True, (msg or "").strip(), ""
    except Exception as e:
        return False, "", f"openai error: {e}"

def _normalize_line(line: str) -> str:
    out = line.strip().strip('"').strip("'").strip("`").strip("*").strip()
    out = re.sub(r"^[-*]\s+", "", out)
    out = MESSAGE_PREFIX_PATTERN.sub("", out).strip()
    out = re.sub(r"\s+", " ", out)
    return out


def _canonicalize_conventional(line: str) -> str:
    match = CONVENTIONAL_HEADER_PATTERN.match(line)
    if not match:
        return line
    commit_type, scope, breaking, subject = match.groups()
    commit_type = commit_type.lower()
    scope = scope or ""
    breaking = breaking or ""
    subject = subject.strip()
    return f"{commit_type}{scope}{breaking}: {subject}"


def _extract_message(text: str, allow_nonconv: bool) -> str:
    candidates: List[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        # Ignore markdown code-fence lines while keeping fenced content.
        if stripped.startswith("```"):
            continue
        line = _normalize_line(raw_line)
        if not line:
            continue
        candidates.append(line)
        if CONVENTIONAL_COMMITS_PATTERN.match(line):
            return _canonicalize_conventional(line)

        embedded = EMBEDDED_CONVENTIONAL_PATTERN.search(line)
        if embedded:
            return _canonicalize_conventional(_normalize_line(embedded.group(0)))

    if allow_nonconv and candidates:
        return candidates[0]
    return ""

def _trim_diff(diff: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0 or len(diff) <= max_chars:
        return diff, False

    marker = "\n\n# ... diff truncated for prompt size ...\n\n"
    budget = max_chars - len(marker)
    if budget <= 200:
        return diff[:max_chars], True

    head = budget // 2
    tail = budget - head
    trimmed = f"{diff[:head]}{marker}{diff[-tail:]}"
    return trimmed, True


def generate_commit_message(diff: str, language: str, force: Optional[str], debug: bool, allow_nonconv: bool) -> str:
    # Always try AI in this order: Ollama → Gemini → OpenAI
    order: List[Provider] = []

    # Respect forced provider if given; otherwise default to Ollama first.
    if force:
        if force == "ollama":
            order.append(Provider("ollama", OLLAMA_MODEL, host=OLLAMA_HOST))
        elif force == "gemini" and GEMINI_KEY and GEMINI_MODEL:
            order.append(Provider("gemini", GEMINI_MODEL, api_key=GEMINI_KEY))
        elif force == "openai" and OPENAI_KEY and OPENAI_MODEL:
            order.append(Provider("openai", OPENAI_MODEL, api_key=OPENAI_KEY))
        else:
            raise RuntimeError(f"forced provider '{force}' unavailable or misconfigured")
    else:
        # Default order (no pre-checks; just try and catch)
        order.append(Provider("ollama", OLLAMA_MODEL, host=OLLAMA_HOST))
        if GEMINI_KEY and GEMINI_MODEL:
            order.append(Provider("gemini", GEMINI_MODEL, api_key=GEMINI_KEY))
        if OPENAI_KEY and OPENAI_MODEL:
            order.append(Provider("openai", OPENAI_MODEL, api_key=OPENAI_KEY))

    if not order:
        raise RuntimeError("no providers configured")

    system_prompt = (
        "You are a commit message generator that strictly follows Conventional Commits format:\n"
        "^type(optional-scope): subject\n"
        f"Allowed types: {', '.join(CONVENTIONAL_TYPES)}.\n"
        "Rules:\n"
        "- Output ONLY the commit message, no code blocks or explanations.\n"
        "- Use imperative mood; concise subject (<= 72 chars ideally).\n"
        f"- Write the message in language: {language}.\n"
        "- If multiple changes, choose the best single type and optional scope.\n"
    )
    user_prompt = f"Generate one commit message for this staged git diff:\n\n{diff}"

    errors: List[str] = []
    for p in order:
        _log(f"trying {p.name} ({p.model})", debug)
        ok, text, err = False, "", ""
        if p.name == "ollama":
            ok, text, err = _ollama_chat(p.model, p.host, system_prompt, user_prompt, debug)
        elif p.name == "gemini":
            ok, text, err = _gemini_complete(p.model, p.api_key, system_prompt, user_prompt)
        elif p.name == "openai":
            ok, text, err = _openai_complete(p.model, p.api_key, system_prompt, user_prompt)

        if err:
            _log(err, debug)
        if not ok and not text:
            errors.append(f"{p.name}: {err or 'request failed'}")
            continue
        text = _extract_message(text, allow_nonconv)

        if not text:
            errors.append(f"{p.name}: empty")
            continue

        if allow_nonconv or CONVENTIONAL_COMMITS_PATTERN.match(text):
            _log(f"{p.name} -> {text}", debug)
            return text

        errors.append(f"{p.name}: non-conventional -> {text!r}")

    raise RuntimeError(" | ".join(errors) if errors else "all providers failed")


# ───────────────────────────────────────────────────────────────────────────────
# Git helpers
# ───────────────────────────────────────────────────────────────────────────────
def _run_git(args: Sequence[str], env: Optional[dict] = None) -> str:
    cmd = ["git", *args]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result.stdout


def is_git_repo() -> bool:
    try:
        out = _run_git(["rev-parse", "--is-inside-work-tree"]).strip()
        return out == "true"
    except subprocess.CalledProcessError:
        return False

def get_diff(diff_per_file: bool) -> str:
    try:
        env = {**os.environ, "GIT_PAGER": "cat"}
        if diff_per_file:
            files = _run_git(["diff", "--cached", "--name-only"], env=env).splitlines()
            chunks = []
            for f in files:
                if not f:
                    continue
                d = _run_git(["diff", "--cached", "--", f], env=env)
                if d.strip():
                    chunks.append(d)
            return "\n".join(chunks)
        return _run_git(["diff", "--cached"], env=env)
    except subprocess.CalledProcessError:
        return ""


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.max_diff_chars < 0:
        _error("--max-diff-chars cannot be negative")
        sys.exit(2)

    if not is_git_repo():
        _error("not a git repository")
        sys.exit(1)

    if args.auto_stage:
        try:
            _run_git(["add", "-u"])
        except subprocess.CalledProcessError as e:
            _error(f"git add -u failed: {(e.output or '').strip()}")
            sys.exit(1)

    diff = get_diff(args.diff_per_file)
    if not diff.strip():
        _error("no staged diff")
        sys.exit(1)

    diff, was_trimmed = _trim_diff(diff, args.max_diff_chars)
    if was_trimmed:
        _log(
            f"staged diff was trimmed to {args.max_diff_chars} chars "
            "for provider prompt stability",
            args.debug,
        )

    try:
        msg = generate_commit_message(
            diff,
            args.language,
            args.provider,
            args.debug,
            args.allow_nonconventional,
        )
    except Exception as e:
        _error(f"generation failed: {e}")
        sys.exit(2)

    msg = _extract_message(msg, args.allow_nonconventional)
    if not msg:
        _error("empty message after sanitize")
        sys.exit(2)

    if not args.allow_nonconventional and not CONVENTIONAL_COMMITS_PATTERN.match(msg):
        _error(f"not conventional: {msg!r}")
        sys.exit(3)

    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
