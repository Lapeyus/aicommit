import re
import subprocess
import sys
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv()

OPENAI_API_KEY= ""
OPENAI_MODEL_NAME = "gpt-3.5-turbo-0125"

GOOGLE_API_KEY = '' 
GEMINI_MODEL_NAME = "gemini-pro"

def get_api_configuration():
    """Determines the API key and model name based on configuration."""
    if GOOGLE_API_KEY and GEMINI_MODEL_NAME:
        return GOOGLE_API_KEY, GEMINI_MODEL_NAME
    elif OPENAI_API_KEY and OPENAI_MODEL_NAME:  
        return OPENAI_API_KEY, OPENAI_MODEL_NAME 
    else:
        print("Please configure either Google Gemini or OpenAI API credentials & model in environment variables.")
        print("(GOOGLE_API_KEY + GEMINI_MODEL_NAME or OPENAI_API_KEY + OPENAI_MODEL_NAME)")
        sys.exit(1)

# Model Configuration 
def configure_service(api_key, model_name):
    """Configures OpenAI or Google Gemini access."""

    if model_name == GEMINI_MODEL_NAME:
        genai.configure(api_key=api_key)
        return True, genai.GenerativeModel(model_name=model_name)
    else:
        return False, OpenAI(api_key=api_key) 

# Message Generation
def generate_commit_message(diff: str, language: str = "en") -> str:
    """Generates a commit message using either OpenAI or Google Gemini."""
    api_key, model_name = get_api_configuration()
    system=f"You are a commit message generator that strictly follows the Conventional Commits specification validated via regex r'^(feat|fix|docs|style|refactor|test|chore)(\([\w-]+\))?: .+$'. "
    user=f"Given the following git diff, suggest a concise and descriptive commit message in {language}: ---- {diff}"
    prompt=system+user

    try:
        use_gemini,service = configure_service(api_key, model_name)

        if use_gemini:
            response = service.generate_content(prompt)
            return response.text
        else:
            completion = service.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],
                temperature=0.7,
                max_tokens=64,
                top_p=1
            )
            return completion.choices[0].message.content.strip().strip('"')

    except Exception as e:
        print(f"Error generating commit message: {e}")
        return ""

def get_diff(diff_per_file: bool = False) -> str:
    try:
        if diff_per_file:
            diff_files = subprocess.check_output(
                "git diff --cached --name-only",
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode("utf-8").strip().split("\n")

            diffs = [
                subprocess.check_output(
                    f"git diff --cached -- {file}",
                    shell=True,
                    stderr=subprocess.STDOUT,
                ).decode("utf-8") for file in diff_files if file
            ]
            return "\n".join(diffs)
        else:
            return subprocess.check_output(
                "git diff --cached",
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error getting git diff: {e.output.decode('utf-8')}")
        return ""

def main():
    try:
        subprocess.check_output("git add -u", shell=True, stderr=subprocess.STDOUT)
        # print("Updated staged files with recent changes.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating staged files: {e.output.decode('utf-8')}")
        sys.exit(1)
        
    try:
        subprocess.check_output("git rev-parse --is-inside-work-tree", shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("This is not a git repository")
        sys.exit(1)

    diff_per_file = "--diff-per-file" in sys.argv  
    commit_language = "en"  

    if "--language" in sys.argv:
        language_index = sys.argv.index("--language")
        commit_language = sys.argv[language_index + 1]

    diff = get_diff(diff_per_file)

    if not diff:
        print("No staged changes found. Make sure there are changes and run `git add .`")
        sys.exit(1)

    if len(diff) > 8000:
        print("The diff is too large to process. Please split your changes into smaller commits.")
        sys.exit(1)

    commit_message = generate_commit_message(diff, commit_language)

    if not commit_message:
        print("No commit message generated.")
        sys.exit(1)

    # Validation (Conventional Commits specification)
    conventional_commits_pattern = r'^(feat|fix|docs|style|refactor|test|chore)(\([\w-]+\))?: .+$'
    if not re.match(conventional_commits_pattern, commit_message):
        print("ERROR: Commit message does not adhere to the Conventional Commits Specification")
        print(f"Commit message: {commit_message}")
        sys.exit(1)

    print(commit_message)
    sys.exit(0)

if __name__ == "__main__":

    main()
    
