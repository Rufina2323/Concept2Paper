import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from src.concept_extractors.base import prepare_texts

logger = logging.getLogger(__name__)


def _extract_single_document_rake(
    text: str, params: Dict[str, Any], top_n: int
) -> Tuple[List[str], List[float]]:
    """
    Extract keywords from a document using multi_rake.

    Args:
        text: Input text string.
        params: RAKE parameters from config.
        top_n: Number of top keywords to return.

    Returns:
        Tuple of (keywords_list, scores_list).
    """
    from multi_rake import Rake

    language = params.get("language", "en")
    min_length = params.get("min_length", 1)
    max_length = params.get("max_length", 4)
    stopwords = params.get("stopwords", None)

    rake = Rake(
        min_chars=min_length,
        max_words=max_length,
        language_code=language,
        stopwords=stopwords,
    )

    keywords_with_scores = rake.apply(text)

    if not keywords_with_scores:
        return [], []

    keywords_with_scores = keywords_with_scores[:top_n]
    concepts = [kw for kw, _ in keywords_with_scores]
    scores = [float(score) for _, score in keywords_with_scores]

    return concepts, scores


def rake_extraction(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract concepts from articles using RAKE.

    RAKE identifies keywords by splitting text at stopword and phrase delimiter positions, then scores candidates using word degree and frequency metrics.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        DataFrame with additional columns: concepts, concept_scores.
    """
    logger.info("Starting RAKE concept extraction")

    # Load parameters
    general = config["general"]
    params = config.get("rake", {})
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

    for idx, text in tqdm(enumerate(texts), total=len(texts), desc="RAKE extraction"):
        if not text or text.strip() == "":
            all_concepts.append([])
            all_scores.append([])
            continue

        try:
            concepts, scores = _extract_single_document_rake(text, params, top_n)
            all_concepts.append(concepts)
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"RAKE failed for document {idx}: {e}")
            all_concepts.append([])
            all_scores.append([])

    df[general["output_raw_concepts_column"]] = all_concepts
    df[general["output_concept_scores_column"]] = all_scores

    logger.info(
        f"RAKE extraction complete. Extracted concepts for {len(df)} documents."
    )
    return df
