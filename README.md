# aicommit

Generate a commit message from staged Git changes using AI providers with Conventional Commit validation.

## Features

- Reads staged diff from `git diff --cached`.
- Tries providers in fallback order: `ollama -> gemini -> openai` (unless forced).
- Normalizes noisy model output (code fences, bullets, prefixes).
- Enforces Conventional Commits by default.
- Supports diff-size trimming for large changes to improve model reliability.
- Prints only the commit subject to stdout, so it can be piped into `git commit -m`.

## Conventional Commit Support

Default allowed types:

- `feat`
- `fix`
- `docs`
- `style`
- `refactor`
- `test`
- `chore`
- `build`
- `ci`
- `perf`
- `revert`

Formats accepted:

- `type: subject`
- `type(scope): subject`
- `type(scope)!: subject`

## Requirements

- Python 3.9+
- Git
- At least one provider configured:
  - Ollama running locally/remotely
  - Gemini API key
  - OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

You can configure providers and behavior via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL_NAME` | `gemma3n:e4b` | Ollama model name |
| `GOOGLE_API_KEY` | _empty_ | Gemini API key |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Gemini model name |
| `OPENAI_API_KEY` | _empty_ | OpenAI API key |
| `OPENAI_MODEL_NAME` | `gpt-3.5-turbo-0125` | OpenAI model name |
| `AICOMMIT_MAX_DIFF_CHARS` | `14000` | Max prompt diff chars (`0` disables truncation) |

## Usage

Basic usage:

```bash
python3 aicommit.py
```

Use it directly in a commit:

```bash
git commit -m "$(python3 aicommit.py)"
```

### CLI Flags

```text
--provider {ollama,gemini,openai}
--language LANGUAGE
--allow-nonconventional
--diff-per-file
--max-diff-chars N
--no-auto-stage
--debug
```

### Examples

Force OpenAI:

```bash
python3 aicommit.py --provider openai
```

Generate in Spanish:

```bash
python3 aicommit.py --language es
```

Disable auto-staging tracked files:

```bash
python3 aicommit.py --no-auto-stage
```

Allow non-Conventional output:

```bash
python3 aicommit.py --allow-nonconventional
```

## Git Alias

Create a message-only alias:

```bash
git config --global alias.aimsg '!python3 /absolute/path/to/aicommit.py'
```

Create an alias that commits with generated message:

```bash
git config --global alias.ai '!f(){ msg="$(python3 /absolute/path/to/aicommit.py)" || exit $?; git commit -m "$msg"; }; f'
```

## Exit Codes

- `0`: success
- `1`: Git context/staging error (not a repo, no staged diff, etc.)
- `2`: generation/provider error
- `3`: output failed Conventional Commit validation

## Troubleshooting

- `not a git repository`: run inside a Git worktree.
- `no staged diff`: stage changes first, or rely on default auto-stage (`git add -u`).
- `forced provider 'X' unavailable or misconfigured`: set that provider's API key/model.
- `not conventional: ...`: rerun, or use `--allow-nonconventional` if intentional.

## License

[LICENSE](LICENSE)
