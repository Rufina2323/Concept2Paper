import logging
import pandas as pd
import yake
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from src.concept_extractors.base import prepare_texts

logger = logging.getLogger(__name__)


def _extract_single_document_yake(
    text: str, params: Dict[str, Any], top_n: int
) -> Tuple[List[str], List[float]]:
    """
    Extract keywords from a single document using YAKE. YAKE scores are inverted — lower scores indicate higher relevance.

    Args:
        text: Input text string.
        params: YAKE parameters from config.
        top_n: Number of top keywords to return.

    Returns:
        Tuple of (keywords_list, scores_list).
    """
    language = params.get("language", "en")
    max_ngram_size = params.get("max_ngram_size", 3)
    dedup_threshold = params.get("deduplication_threshold", 0.9)
    dedup_algo = params.get("deduplication_algo", "seqm")
    window_size = params.get("window_size", 1)
    num_keywords = params.get("num_of_keywords", top_n)

    extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=dedup_threshold,
        dedupFunc=dedup_algo,
        windowsSize=window_size,
        top=num_keywords,
    )

    keywords_with_scores = extractor.extract_keywords(text)

    if not keywords_with_scores:
        return [], []

    keywords_with_scores = keywords_with_scores[:top_n]
    concepts = [kw for kw, _ in keywords_with_scores]
    scores = [float(score) for _, score in keywords_with_scores]

    return concepts, scores


def yake_extraction(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract concepts from articles using YAKE. YAKE uses statistical text features (casing, word position, frequency, relatedness to context, dispersion) to score candidate keywords.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        DataFrame with additional columns: concepts, concept_scores.
    """
    logger.info("Starting YAKE concept extraction")

    # Load parameters
    general = config["general"]
    params = config.get("yake", {})
    global_top_n = general.get("top_n", 10)
    top_n = params.get("top_n", global_top_n)

    # Read input data
    input_file = general["input_file"]
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} documents from {input_file}")

    # Prepare texts
    texts = prepare_texts(df, config)

    # Extract keywords for each document
    all_concepts = []
    all_scores = []

    for idx, text in tqdm(enumerate(texts), total=len(texts), desc="YAKE extraction"):
        if not text or text.strip() == "":
            all_concepts.append([])
            all_scores.append([])
            continue

        try:
            concepts, scores = _extract_single_document_yake(text, params, top_n)
            all_concepts.append(concepts)
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"YAKE failed for document {idx}: {e}")
            all_concepts.append([])
            all_scores.append([])

    df[general["output_raw_concepts_column"]] = all_concepts
    df[general["output_concept_scores_column"]] = all_scores

    logger.info(
        f"YAKE extraction complete. Extracted concepts for {len(df)} documents."
    )
    return df
