#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# Optional providers
try:
    import ollama as ollama_pkg  # type: ignore
except Exception:
    ollama_pkg = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

load_dotenv()

# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────
CONVENTIONAL_COMMITS_PATTERN = re.compile(
    r'^(feat|fix|docs|style|refactor|test|chore)(\([\w-]+\))?: .+$'
)

# Ollama (default)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "gemma3n:e4b")

# Gemini
GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-0125")


# ───────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ───────────────────────────────────────────────────────────────────────────────
def _has_flag(argv: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)

def _read_arg(argv: List[str], name: str, default: Optional[str] = None) -> Optional[str]:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    for a in argv:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default

def _log(msg: str, debug: bool):
    if debug:
        sys.stderr.write(f"[commit-gen] {msg}\n")
        sys.stderr.flush()


# ───────────────────────────────────────────────────────────────────────────────
# Providers (always try AI; no non-AI fallback)
# ───────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Provider:
    name: str  # "ollama" | "gemini" | "openai"
    model: str
    api_key: str = ""
    host: str = ""  # for ollama

def _ollama_chat(model: str, host: str, system_prompt: str, user_prompt: str, debug: bool) -> Tuple[bool, str, str]:
    try:
        # Prefer python client if present
        if ollama_pkg is not None:
            # Ensure host is used by the client
            if host:
                os.environ["OLLAMA_HOST"] = host
            resp = ollama_pkg.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.2, "num_predict": 128},
            )
            content = (resp.get("message") or {}).get("content", "")
            return True, (content or "").strip(), ""
        # Raw HTTP fallback
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2, "num_predict": 128},
        }
        req = Request(
            f"{host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        text = (data.get("message") or {}).get("content", "")
        return True, (text or "").strip(), ""
    except (URLError, HTTPError) as e:
        return False, "", f"ollama network error: {e}"
    except Exception as e:
        return False, "", f"ollama error: {e}"

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
            return False, "", "openai sdk missing"
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

def _sanitize_first_line(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        # drop code fences if any
        parts = text.split("```")
        text = "".join(parts[1:-1]) if len(parts) >= 3 else parts[-1]
        text = text.strip()
    # return first non-empty line only
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""


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

    system_prompt = (
        "You are a commit message generator that strictly follows Conventional Commits format:\n"
        "^(feat|fix|docs|style|refactor|test|chore)(\\([\\w-]+\\))?: .+$\n"
        "Rules:\n"
        "• Output ONLY the commit message, no code blocks or explanations.\n"
        "• Use imperative mood; concise subject (≤ 72 chars ideally).\n"
        f"• Write the message in language: {language}.\n"
        "• If multiple changes, choose the best single type and optional scope.\n"
    )

    errors: List[str] = []
    for p in order:
        _log(f"trying {p.name} ({p.model})", debug)
        ok, text, err = False, "", ""
        if p.name == "ollama":
            ok, text, err = _ollama_chat(p.model, p.host, system_prompt, diff, debug)
        elif p.name == "gemini":
            ok, text, err = _gemini_complete(p.model, p.api_key, system_prompt, diff)
        elif p.name == "openai":
            ok, text, err = _openai_complete(p.model, p.api_key, system_prompt, diff)

        if err:
            _log(err, debug)
        text = _sanitize_first_line(text)

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
def is_git_repo() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False

def get_diff(diff_per_file: bool, debug: bool) -> str:
    try:
        env = {**os.environ, "GIT_PAGER": "cat"}
        if diff_per_file:
            files = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"], stderr=subprocess.STDOUT, text=True, env=env
            ).splitlines()
            chunks = []
            for f in files:
                if not f:
                    continue
                d = subprocess.check_output(
                    ["git", "diff", "--cached", "--", f], stderr=subprocess.STDOUT, text=True, env=env
                )
                if d.strip():
                    chunks.append(d)
            return "\n".join(chunks)
        return subprocess.check_output(["git", "diff", "--cached"], stderr=subprocess.STDOUT, text=True, env=env)
    except subprocess.CalledProcessError:
        return ""


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    debug = _has_flag(sys.argv, "--debug")
    force = _read_arg(sys.argv, "--provider", None)  # ollama|gemini|openai
    allow_nonconv = _has_flag(sys.argv, "--allow-nonconventional")
    diff_per_file = _has_flag(sys.argv, "--diff-per-file")
    language = _read_arg(sys.argv, "--language", "en") or "en"

    if not is_git_repo():
        _log("not a git repo", debug)
        sys.exit(1)

    try:
        subprocess.check_output(["git", "add", "-u"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        _log(f"git add failed: {e}", debug)
        sys.exit(1)

    diff = get_diff(diff_per_file, debug)
    if not diff.strip():
        _log("no staged diff", debug)
        sys.exit(1)

    try:
        msg = generate_commit_message(diff, language, force, debug, allow_nonconv)
    except Exception as e:
        _log(f"generation failed: {e}", True)
        sys.exit(2)

    msg = _sanitize_first_line(msg)
    if not msg:
        _log("empty after sanitize", debug)
        sys.exit(2)

    if not allow_nonconv and not CONVENTIONAL_COMMITS_PATTERN.match(msg):
        _log(f"not conventional: {msg!r}", debug)
        sys.exit(3)

    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
