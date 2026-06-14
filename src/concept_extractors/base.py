import pandas as pd
from typing import Any, Dict


def prepare_texts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    Default text preparation from dataframe based on config settings.

    Args:
        df: DataFrame with title and abstract columns.
        config: Full configuration dictionary.

    Returns:
        Series of text strings ready for processing.
    """
    general = config["general"]
    title_col = general["title_column"]
    abstract_col = general["abstract_column"]
    combine = general.get("combine_title_abstract", True)
    separator = general.get("separator", " . ")

    if combine:
        texts = (
            df[title_col].fillna("").astype(str)
            + separator
            + df[abstract_col].fillna("").astype(str)
        )
    else:
        texts = df[abstract_col].fillna("").astype(str)

    return texts
