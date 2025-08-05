import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = "gpt-3.5-turbo-0125"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.0-flash" 

CONVENTIONAL_COMMITS_PATTERN = re.compile(r'^(feat|fix|docs|style|refactor|test|chore)(\([\w-]+\))?: .+$')


@dataclass(frozen=True)
class APIConfig:
    api_key: str
    model_name: str


def get_api_configuration() -> APIConfig:
    if GOOGLE_API_KEY and GEMINI_MODEL_NAME:
        return APIConfig(GOOGLE_API_KEY, GEMINI_MODEL_NAME)
    if OPENAI_API_KEY and OPENAI_MODEL_NAME:
        return APIConfig(OPENAI_API_KEY, OPENAI_MODEL_NAME)
    raise RuntimeError("No valid API configuration found")


def configure_service(config: APIConfig):
    if config.model_name == GEMINI_MODEL_NAME:
        genai.configure(api_key=config.api_key)
        return True, genai.GenerativeModel(model_name=config.model_name)
    return False, OpenAI(api_key=config.api_key)


def generate_commit_message(diff: str, language: str = "en") -> str:
    api_config = get_api_configuration()
    use_gemini, service = configure_service(api_config)

    system_prompt = (
        "You are a commit message generator that strictly follows Conventional Commits "
        "format: ^(feat|fix|docs|style|refactor|test|chore)(\\([\\w-]+\\))?: .+$."
        " Generate ONLY the commit message without any additional text."
    )

    if use_gemini:
        response = service.generate_content(system_prompt + diff)
        return response.text.strip()
    completion = service.chat.completions.create(
        model=api_config.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diff},
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1,
    )
    return completion.choices[0].message.content.strip('"').strip()


def get_diff(diff_per_file: bool = False) -> str:
    try:
        if diff_per_file:
            diff_files = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"],
                stderr=subprocess.STDOUT,
                text=True,
            ).strip().split("\n")

            diffs = [
                subprocess.check_output(
                    ["git", "diff", "--cached", "--", file],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for file in diff_files if file
            ]
            return "\n".join(diffs)
        return subprocess.check_output(
            ["git", "diff", "--cached"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""


def is_git_repository() -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    if not is_git_repository():
        sys.exit(1)

    try:
        subprocess.check_output(["git", "add", "-u"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        sys.exit(1)

    diff_per_file = "--diff-per-file" in sys.argv
    commit_language = "en"
    if "--language" in sys.argv:
        idx = sys.argv.index("--language")
        if idx + 1 < len(sys.argv):
            commit_language = sys.argv[idx + 1]

    diff = get_diff(diff_per_file)
    if not diff:
        sys.exit(1)

    commit_message = generate_commit_message(diff, commit_language)
    if not commit_message or not CONVENTIONAL_COMMITS_PATTERN.match(commit_message):
        sys.exit(1)

    print(commit_message)
    sys.exit(0)


if __name__ == "__main__":
    main()
