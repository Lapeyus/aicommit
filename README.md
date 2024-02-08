# AI-Powered Commit Message Automation

Harness the power of AI to streamline your Git commit messages with this Python script. By leveraging both OpenAI's ChatGPT and Google's Generative AI, automate the crafting of insightful and standard-compliant commit messages. Dive into a solution that combines the best of AI technology to enhance your software development workflow.

## Introduction

Crafting precise and insightful commit messages is crucial for efficient project management and seamless team collaboration. This guide introduces a Python-driven solution to automate commit message generation, integrating AI models from OpenAI and Google. Embrace flexibility and best practices in your development process.

## Problem Statement

Manual composition of commit messages is time-consuming and monotonous. This solution aims to automate this process, delivering informative, standardized messages, and allowing developers to focus on coding.

## Solution Overview

The script provides an interface to OpenAI's ChatGPT and Google's Generative AI, selecting the best AI provider based on available credentials. It includes environment configuration, AI service initialization, and commit message generation following the Conventional Commits specification.

## Technical Details

```python
# Python code snippet for API configuration and service initialization
def get_api_configuration():
    # Logic to determine which API keys and model names are configured

def configure_service(api_key, model_name):
    # Setup for either OpenAI or Google Generative AI model

def generate_commit_message(diff: str, language: str = "en"):
    # Generates commit message using the configured AI service
    # Detailed prompt crafting and AI model response handling

def validate_and_format_message(message: str):
    # Validates the generated commit message against Conventional Commits
```

## Usage

To use this script set the API key your model, either `OPENAI_API_KEY` or `GOOGLE_API_KEY`  and save the file `~/.aicommit.py`, install the requirements with `pip install -r requirements` and then configure a Git alias as follows:
```bash
git config --global alias.ai '!git commit -am "$(python ~/.aicommit.py)"'
```
Invoke with `git ai` to automatically generate and commit messages based on staged changes, ensuring consistency and adherence to best practices.

## Benefits

- **Time Efficiency**: Reduces effort in crafting commit messages.
- **Consistency and Compliance**: Ensures uniform adherence to standards.
- **Enhanced Project Log**: Provides a cleaner, more informative commit history.

## Customization and Exploration

This script encourages customization and exploration. Adapt it to fit team-specific guidelines or explore alternative AI models for improved outcomes.

## Collaboration Invitation

Join the development of this AI-powered commit message generation solution. Share insights, contribute, and explore the potential of intelligent automation in software development.

Embrace AI to optimize your development workflows, merging technology and creativity for efficient software development.
 
