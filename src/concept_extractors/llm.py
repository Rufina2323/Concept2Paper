import logging
import os
import time
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_texts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Prepare text input from dataframe."""
    general = config["general"]
    llm_params = config.get("llm", {})
    title_col = general["title_column"]
    abstract_col = general["abstract_column"]
    combine = general.get("combine_title_abstract", True)
    user_prompt_template = llm_params.get(
        "user_prompt_template",
        "{title}\n{abstract}",
    )

    if combine:
        texts = df.apply(
            lambda row: user_prompt_template.format(
                title=str(row[title_col] or ""),
                abstract=str(row[abstract_col] or ""),
            ),
            axis=1,
        )
    else:
        texts = df[abstract_col].fillna("").astype(str)

    return texts


def _extract_openai(
    params: Dict[str, Any], system_prompt: str, user_prompt: str
) -> str:
    """
    Extract keywords using OpenAI API.

    Args:
        text: Input text.
        params: OpenAI-specific parameters.
        prompt_template: Prompt template with {text} placeholder.

    Returns:
        Raw response text from the model.
    """
    from openai import OpenAI

    api_key = os.environ.get(params.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise ValueError(
            f"OpenAI API key not found in environment variable: "
            f"{params.get('api_key_env', 'OPENAI_API_KEY')}"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=params.get("model", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=params.get("temperature", 0.0),
        extra_body={"reasoning": {"enabled": params.get("reasoning", True)}},
    )

    return response.choices[0].message.content


def llm_extraction(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract concepts from articles using a Large Language Model.

    Supported provider: OpenAI (GPT-4, GPT-3.5, etc.)

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        DataFrame with additional columns: concepts.
    """
    logger.info("Starting LLM concept extraction")

    # Load parameters
    general = config["general"]
    llm_params = config.get("llm", {})
    global_top_n = general.get("top_n", 10)
    top_n = llm_params.get("top_n", global_top_n)

    provider = llm_params.get("provider", "openai")
    provider_params = llm_params.get(provider, {})
    system_prompt = llm_params.get(
        "system_prompt_template",
        "Extract the most important keywords from text.",
    ).format(top_n=top_n)

    batch_size = llm_params.get("batch_size", 5)
    delay_between_batches = llm_params.get("delay_between_batches", 1.0)
    max_retries = llm_params.get("max_retries", 3)
    retry_delay = llm_params.get("retry_delay", 5.0)

    # Select extraction function based on provider
    extract_fn_map = {
        "openai": _extract_openai,
    }

    if provider not in extract_fn_map:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: {list(extract_fn_map.keys())}"
        )

    extract_fn = extract_fn_map[provider]
    logger.info(f"Using LLM provider: {provider}")

    # Read input data
    input_file = general["input_file"]
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} documents from {input_file}")

    # Prepare texts
    texts = prepare_texts(df, config)

    # Extract keywords for each document
    all_concepts = []

    for idx, text in tqdm(
        enumerate(texts), total=len(texts), desc=f"LLM ({provider}) extraction"
    ):
        if not text or text.strip() == "":
            all_concepts.append([])
            continue

        # Retry logic
        success = False
        for attempt in range(max_retries):
            try:
                raw_response = extract_fn(provider_params, system_prompt, text)
                concepts = raw_response.split("\n")
                all_concepts.append(concepts)
                success = True
                break
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed for document {idx} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if not success:
            logger.error(
                f"LLM extraction failed for document {idx} after {max_retries} attempts"
            )
            all_concepts.append([])

        # Rate limiting: pause between batches
        if (idx + 1) % batch_size == 0 and idx < len(texts) - 1:
            time.sleep(delay_between_batches)

    df[general["output_raw_concepts_column"]] = all_concepts

    logger.info(f"LLM extraction complete. Extracted concepts for {len(df)} documents.")
    return df
