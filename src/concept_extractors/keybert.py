import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from src.concept_extractors.base import prepare_texts
from keybert import KeyBERT

logger = logging.getLogger(__name__)


def _extract_single_document_keybert(
    text: str,
    kw_model,
    params: Dict[str, Any],
    top_n: int,
) -> Tuple[List[str], List[float]]:
    """
    Extract keywords from a document using KeyBERT.

    Args:
        text: Input text string.
        kw_model: Pre-loaded KeyBERT model.
        params: KeyBERT parameters from config.
        top_n: Number of top keywords to return.

    Returns:
        Tuple of (keywords_list, scores_list).
    """
    ngram_range = tuple(params.get("keyphrase_ngram_range", [1, 3]))
    stop_words = params.get("stop_words", "english")
    if stop_words == "null" or stop_words is None:
        stop_words = None

    use_maxsum = params.get("use_maxsum", False)
    use_mmr = params.get("use_mmr", True)
    diversity = params.get("diversity", 0.7)
    nr_candidates = params.get("nr_candidates", 20)
    seed_keywords = params.get("seed_keywords", None)
    highlight = params.get("highlight", False)

    kwargs = {
        "docs": text,
        "keyphrase_ngram_range": ngram_range,
        "stop_words": stop_words,
        "top_n": top_n,
        "highlight": highlight,
    }

    if seed_keywords:
        kwargs["seed_keywords"] = seed_keywords

    if use_mmr:
        kwargs["use_mmr"] = True
        kwargs["diversity"] = diversity
    elif use_maxsum:
        kwargs["use_maxsum"] = True
        kwargs["nr_candidates"] = nr_candidates

    keywords_with_scores = kw_model.extract_keywords(**kwargs)

    if not keywords_with_scores:
        return [], []

    concepts = [kw for kw, _ in keywords_with_scores]
    scores = [float(score) for _, score in keywords_with_scores]

    return concepts, scores


def keybert_extraction(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract concepts from articles using KeyBERT. KeyBERT creates document and word embeddings with sentence-transformers, then uses cosine similarity to find the most document-representative keywords.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        DataFrame with additional columns: concepts, concept_scores.
    """
    logger.info("Starting KeyBERT concept extraction")

    # Load parameters
    general = config["general"]
    params = config.get("keybert", {})
    global_top_n = general.get("top_n", 10)
    top_n = params.get("top_n", global_top_n)
    model_name = params.get("model_name", "all-MiniLM-L6-v2")

    # Read input data
    input_file = general["input_file"]
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} documents from {input_file}")

    # Prepare texts
    texts = prepare_texts(df, config)

    # Load model
    kw_model = KeyBERT(model=model_name)

    # Extract keywords for each document
    all_concepts = []
    all_scores = []

    for idx, text in tqdm(
        enumerate(texts), total=len(texts), desc="KeyBERT extraction"
    ):
        if not text or text.strip() == "":
            all_concepts.append([])
            all_scores.append([])
            continue

        try:
            concepts, scores = _extract_single_document_keybert(
                text, kw_model, params, top_n
            )
            all_concepts.append(concepts)
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"KeyBERT failed for document {idx}: {e}")
            all_concepts.append([])
            all_scores.append([])

    df[general["output_raw_concepts_column"]] = all_concepts
    df[general["output_concept_scores_column"]] = all_scores

    logger.info(
        f"KeyBERT extraction complete. Extracted concepts for {len(df)} documents."
    )
    return df
