import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

from src.concept_extractors.rake import rake_extraction
from src.concept_extractors.yake import yake_extraction
from src.concept_extractors.keybert import keybert_extraction
from src.concept_extractors.llm import llm_extraction
from src.utils import load_config
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_results(result: pd.DataFrame, output_dir: str, file_name: str) -> None:
    """
    Save extraction results to CSV.

    The 'concepts' and 'concept_scores' columns (which are lists) are stored as string representations in the CSV.

    Args:
        result: DataFrame with extraction results.
        output_dir: Path to the output directory.
        file_name: Name of the output CSV file.
    """

    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def extract_concepts(
    input_file: str,
    output_dir: str,
    methods: list[str],
    title_column: str = "title",
    abstract_column: str = "abstract",
    output_raw_concepts_column: str = "raw_concepts",
    output_concept_scores_column: str = "concept_scores",
    combine_title_abstract: bool = True,
    separator: str = " . ",
    top_n: int = 10,
    rake_config: Optional[dict] = None,
    yake_config: Optional[dict] = None,
    keybert_config: Optional[dict] = None,
    llm_config: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Main concept extraction pipeline.

    Args:
        input_file: Path to input CSV with title/abstract columns.
        output_dir: Directory to save results.
        methods: List of extraction methods to run.
        title_column: Name of the title column.
        abstract_column: Name of the abstract column.
        output_raw_concepts_column: Name for raw concepts output column.
        output_concept_scores_column: Name for concept scores output column.
        id_column: Name of the ID column.
        combine_title_abstract: Whether to concatenate title + abstract.
        separator: Separator when combining title and abstract.
        top_n: Default number of top concepts per document.
        rake_config: Parameters specific to RAKE method.
        yake_config: Parameters specific to YAKE method.
        keybert_config: Parameters specific to KeyBERT method.
        llm_config: Parameters specific to LLM method.

    Returns:
        Dict mapping method name to its extracted concepts DataFrame.
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    general_params = {
        "input_file": input_file,
        "title_column": title_column,
        "abstract_column": abstract_column,
        "output_raw_concepts_column": output_raw_concepts_column,
        "output_concept_scores_column": output_concept_scores_column,
        "combine_title_abstract": combine_title_abstract,
        "separator": separator,
        "top_n": top_n,
    }

    extraction_registry = {
        "rake": (rake_extraction, rake_config or {}),
        "yake": (yake_extraction, yake_config or {}),
        "keybert": (keybert_extraction, keybert_config or {}),
        "llm": (llm_extraction, llm_config or {}),
    }

    results = {}

    for method in methods:
        logger.info(f"Running concept extraction: {method}")

        if method not in extraction_registry:
            raise ValueError(
                f"Unknown extraction method: '{method}'. "
                f"Available: {list(extraction_registry.keys())}"
            )

        extract_fn, method_config = extraction_registry[method]
        params = {"general": general_params, "method": method_config}

        concepts = extract_fn(params)

        file_name = f"{method}/raw_concepts.csv"
        save_results(concepts, output_dir, file_name)

        results[method] = concepts

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract concepts/keywords from scientific articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/concept_extracting.yaml",
        help="Path to YAML configuration file (default: configs/concept_extracting.yaml)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        extract_concepts(
            input_file=config["general"]["input_file"],
            output_dir=config["general"]["output_dir"],
            title_column=config["general"].get("title_column", "title"),
            abstract_column=config["general"].get("abstract_column", "abstract"),
            output_raw_concepts_column=config["general"].get(
                "output_raw_concepts_column", "raw_concepts"
            ),
            output_concept_scores_column=config["general"].get(
                "output_concept_scores_column", "concept_scores"
            ),
            combine_title_abstract=config["general"].get(
                "combine_title_abstract", True
            ),
            separator=config["general"].get("separator", " . "),
            methods=config["concept_extraction"]["methods"],
            top_n=config["concept_extraction"].get("top_n", 10),
            rake_config=config.get("rake"),
            yake_config=config.get("yake"),
            keybert_config=config.get("keybert"),
            llm_config=config.get("llm"),
        )
        logger.info("Concept extraction completed successfully!")
    except Exception as e:
        logger.error(f"Concept extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
