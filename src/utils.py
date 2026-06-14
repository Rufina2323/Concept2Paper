import yaml
from typing import Any, Dict, List
import ast
import os


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_concept_list(val) -> List[str]:
    return [str(concept).strip() for concept in ast.literal_eval(val) if concept]


def call_claude(prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_gpt(prompt: str, max_tokens: int) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
