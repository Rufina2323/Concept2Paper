import argparse
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.utils import load_config, parse_concept_list


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


EXTRA_STOPWORDS: Set[str] = {
    "using",
    "based",
    "use",
    "used",
    "approach",
    "novel",
    "new",
    "improved",
    "propose",
    "proposed",
    "technique",
    "techniques",
    "method",
    "methods",
    "study",
    "analysis",
    "paper",
    "work",
    "result",
    "results",
    "show",
    "demonstrate",
    "present",
    "investigate",
    "explore",
    "evaluate",
    "experiment",
    "experimental",
    "performance",
    "effective",
    "efficient",
    "also",
    "various",
    "different",
    "several",
    "many",
}

PROTECTED_WORDS: Set[str] = {
    "deep",
    "machine",
    "neural",
    "network",
    "networks",
    "learning",
    "natural",
    "language",
    "processing",
    "computer",
    "vision",
    "reinforcement",
    "transfer",
    "federated",
    "generative",
    "adversarial",
    "attention",
    "self",
    "supervised",
    "unsupervised",
    "semi",
    "few",
    "shot",
    "zero",
    "pre",
    "fine",
    "training",
    "inference",
    "classification",
    "detection",
    "segmentation",
    "recognition",
    "generation",
    "extraction",
    "embedding",
    "representation",
    "knowledge",
    "graph",
    "ontology",
    "model",
    "optimization",
    "gradient",
    "loss",
    "function",
    "distribution",
    "latent",
    "feature",
    "layer",
    "encoder",
    "decoder",
    "diffusion",
    "contrastive",
    "metric",
    "reward",
    "policy",
    "agent",
    "state",
    "action",
    "environment",
    "image",
    "text",
    "speech",
    "audio",
    "video",
    "point",
    "cloud",
    "sequence",
    "token",
    "prompt",
    "retrieval",
    "memory",
    "reasoning",
    "alignment",
    "robustness",
    "fairness",
    "privacy",
    "safety",
    "bias",
    "calibration",
}

DEFAULT_ACRONYMS: Dict[str, str] = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "gru": "gated recurrent unit",
    "gpt": "generative pre-trained transformer",
    "bert": "bidirectional encoder representations from transformers",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "pca": "principal component analysis",
    "gan": "generative adversarial network",
    "vae": "variational autoencoder",
    "rl": "reinforcement learning",
    "rlhf": "reinforcement learning from human feedback",
    "dpo": "direct preference optimization",
    "ner": "named entity recognition",
    "pos": "part of speech",
    "iot": "internet of things",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "tfidf": "term frequency inverse document frequency",
    "lda": "latent dirichlet allocation",
    "hmm": "hidden markov model",
    "sgd": "stochastic gradient descent",
    "auc": "area under the curve",
    "rmse": "root mean square error",
    "mri": "magnetic resonance imaging",
    "fmri": "functional magnetic resonance imaging",
    "eeg": "electroencephalography",
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "elbo": "evidence lower bound",
    "kl": "kullback leibler",
    "mcmc": "markov chain monte carlo",
    "mlp": "multilayer perceptron",
    "vit": "vision transformer",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[" "„‟]", '"', text)
    text = re.sub(r"[''‛]", "'", text)
    text = re.sub(r"[–—―]", "-", text)
    text = re.sub(r"[^\w\s\-/().+#&]", "", text)
    text = re.sub(r"^\s*[-•·]\s*", "", text)
    text = re.sub(r"\s*\(\s*\)\s*", "", text)
    return text.strip(" -.,;:")


def expand_acronym(text: str) -> str:
    match = re.match(r"^(.+?)\s*\(([a-zA-Z\-]+)\)\s*$", text)
    if match:
        text = match.group(1).strip()
    return DEFAULT_ACRONYMS.get(text.lower().replace(" ", "").replace("-", ""), text)


def load_spacy(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        logger.info(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        return spacy.load(model_name, disable=["parser", "ner"])


def lemmatize(text: str, nlp) -> str:
    if not text or len(text) < 2:
        return text
    doc = nlp(text.lower())
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower()
        word = tok.text.lower()
        is_stop = tok.is_stop or lemma in EXTRA_STOPWORDS or word in EXTRA_STOPWORDS
        is_protected = word in PROTECTED_WORDS or lemma in PROTECTED_WORDS
        if is_stop and not is_protected and len(doc) > 1:
            continue
        # Protected words keep their surface form, others get lemmatized
        if "-" in tok.text:
            tokens.append(word)
        else:
            tokens.append(lemma)
    result = re.sub(r"^(the|a|an|of|for|in|on|to|with|and|or)\s+", "", " ".join(tokens))
    return result.strip()


def preprocess_concept(concept: str, nlp) -> str:
    return expand_acronym(lemmatize(clean_text(concept), nlp)).strip()


def make_state() -> Dict[str, Any]:
    return {
        "concept_to_canonical": {},
        "canonical_embeddings": {},
        "cluster_members": defaultdict(set),
        "canonical_to_cid": {},
        "next_cluster_id": 0,
    }


def load_state(path: Optional[str]) -> Dict[str, Any]:
    if path and Path(path).exists():
        with open(path, "rb") as f:
            state = pickle.load(f)
        n_concepts = len(state["concept_to_canonical"])
        n_clusters = len(state["canonical_embeddings"])
        logger.info(
            f"State loaded: {n_concepts} concepts, {n_clusters} clusters <- {path}"
        )
        return state
    logger.info("No existing state found — starting fresh")
    return make_state()


def save_state(state: Dict[str, Any], path: Optional[str]):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_concepts = len(state["concept_to_canonical"])
    n_clusters = len(state["canonical_embeddings"])
    logger.info(f"State saved: {n_concepts} concepts, {n_clusters} clusters -> {path}")


def add_to_cluster(state: Dict, concept: str, canonical: str, embedding: np.ndarray):
    state["concept_to_canonical"][concept] = canonical
    state["cluster_members"][canonical].add(concept)
    if canonical not in state["canonical_embeddings"]:
        state["canonical_embeddings"][canonical] = embedding
        state["canonical_to_cid"][canonical] = state["next_cluster_id"]
        state["next_cluster_id"] += 1


def get_canonical_matrix(state: Dict):
    canonicals = list(state["canonical_embeddings"].keys())
    matrix = np.stack([state["canonical_embeddings"][c] for c in canonicals])
    return canonicals, matrix


def cluster_concepts(
    concepts: List[str],
    embeddings: np.ndarray,
    state: Dict,
    threshold: float,
):
    n_clusters = len(state["canonical_embeddings"])

    # Try assigning to existing clusters first
    if n_clusters > 0:
        canonicals, matrix = get_canonical_matrix(state)
        sim_matrix = cosine_similarity(embeddings, matrix)
        unassigned = []
        for i, concept in tqdm(
            enumerate(concepts), total=len(concepts), desc="Assigning to existing"
        ):
            best_idx = int(np.argmax(sim_matrix[i]))
            if sim_matrix[i, best_idx] >= threshold:
                add_to_cluster(state, concept, canonicals[best_idx], embeddings[i])
            else:
                unassigned.append(i)
        assigned = len(concepts) - len(unassigned)
        logger.info(f"Assigned {assigned} to existing, {len(unassigned)} unassigned")
        concepts = [concepts[i] for i in unassigned]
        embeddings = (
            embeddings[unassigned] if unassigned else np.empty((0, embeddings.shape[1]))
        )

    # Greedy clustering for the rest
    for i, concept in tqdm(
        enumerate(concepts), total=len(concepts), desc="Greedy clustering"
    ):
        emb = embeddings[i]
        if state["canonical_embeddings"]:
            cans, mat = get_canonical_matrix(state)
            sims = cosine_similarity(emb.reshape(1, -1), mat)[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= threshold:
                add_to_cluster(state, concept, cans[best_idx], emb)
                continue
        add_to_cluster(state, concept, concept, emb)


def merge_singular_plural(state: Dict):
    canonical_set = set(state["canonical_embeddings"])
    merges = {
        c: c[:-1] for c in canonical_set if c.endswith("s") and c[:-1] in canonical_set
    }

    for old, new in merges.items():
        members = state["cluster_members"].pop(old, set())
        state["cluster_members"][new].update(members)
        for m in members:
            state["concept_to_canonical"][m] = new
        state["canonical_to_cid"].pop(old, None)
        state["canonical_embeddings"].pop(old, None)

    if merges:
        logger.info(f"Merged {len(merges)} singular/plural pairs")


def preprocess_raw_concepts(all_raw: Set[str], nlp) -> Dict[str, str]:
    logger.info("Preprocessing concepts...")
    raw_to_pp = {r: preprocess_concept(r, nlp) for r in all_raw}
    return {r: p for r, p in raw_to_pp.items() if len(p) >= 2}


def preprocess_all_concepts(
    df: pd.DataFrame, raw_col: str, nlp
) -> Tuple[pd.Series, Dict[str, str]]:
    logger.info("Parsing concept lists...")
    concept_lists = df[raw_col].map(parse_concept_list)

    all_raw = {c for lst in concept_lists for c in lst if c}
    logger.info(f"Total unique raw concepts: {len(all_raw)}")

    logger.info("Preprocessing concepts...")
    raw_to_pp = {r: preprocess_concept(r, nlp) for r in all_raw}
    raw_to_pp = {r: p for r, p in raw_to_pp.items() if len(p) >= 2}
    logger.info(f"Unique preprocessed concepts: {len(set(raw_to_pp.values()))}")

    return concept_lists, raw_to_pp


def process_new_concepts(
    unique_pp: Set[str],
    state: Dict,
    encoder: SentenceTransformer,
    threshold: float,
    batch_size: int,
):
    known_set = state["concept_to_canonical"]
    new_concepts = [c for c in unique_pp if c not in known_set]
    known = len(unique_pp) - len(new_concepts)
    logger.info(f"Already known: {known}, new: {len(new_concepts)}")

    if not new_concepts:
        return

    logger.info(f"Encoding {len(new_concepts)} new concepts...")
    embeddings = encoder.encode(
        new_concepts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info("Clustering new concepts...")
    cluster_concepts(new_concepts, embeddings, state, threshold)
    merge_singular_plural(state)


def build_output(
    concept_lists: pd.Series,
    raw_to_pp: Dict[str, str],
    state: Dict,
) -> Tuple[List[list], List[list], List[list]]:
    normalized, cluster_ids, preprocessed = [], [], []

    for concept_list in concept_lists:
        norm_row, cid_row, prep_row = [], [], []
        for raw in concept_list:
            pp = raw_to_pp.get(raw)
            if not pp or len(pp) < 2:
                norm_row.append(None)
                cid_row.append(-1)
                prep_row.append(None)
                continue
            canonical = state["concept_to_canonical"].get(pp)
            cid = state["canonical_to_cid"].get(canonical, -1) if canonical else -1
            norm_row.append(canonical)
            cid_row.append(cid)
            prep_row.append(pp)
        normalized.append(norm_row)
        cluster_ids.append(cid_row)
        preprocessed.append(prep_row)

    return normalized, cluster_ids, preprocessed


def build_cluster_table(state: Dict) -> pd.DataFrame:
    rows = [
        {
            "cluster_id": state["canonical_to_cid"].get(canon, -1),
            "canonical_concept": canon,
            "member_concept": member,
        }
        for canon, members in state["cluster_members"].items()
        for member in sorted(members)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["cluster_id", "canonical_concept", "member_concept"])
        .reset_index(drop=True)
    )


def normalize_concepts(
    df: pd.DataFrame,
    *,
    raw_concepts_col: str = "raw_concepts",
    year_col: str = "year",
    target_year: Optional[int] = None,
    similarity_threshold: float = 0.85,
    embedding_model: str = "all-MiniLM-L6-v2",
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 256,
    return_details: bool = False,
    state_file: Optional[str] = None,
    save: bool = False,
    normalized_concepts_column: str = "normalized_concepts",
    cluster_ids_column: str = "cluster_ids",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Normalize concepts in a DataFrame.

    Args:
        df : DataFrame with raw concepts.
        raw_concepts_col : column containing raw concept lists.
        year_col : column with publication year.
        target_year : if set, concepts before this year are processed first
                    to build vocabulary, then target-year concepts are
                    normalized against it.
        similarity_threshold : cosine similarity threshold for clustering.
        embedding_model : sentence-transformers model name.
        spacy_model : spaCy model for lemmatization.
        batch_size : encoding batch size.
        return_details : include preprocessed concepts column.
        state_file : path to load/save state for incremental runs.
        save : persist state after normalization.
        normalized_concepts_column : column name with normalized concepts.
        cluster_ids_column : column name with cluster ids for ormalized concepts.

    Returns:
        (df_normalized, cluster_table, state)
    """
    df = df.copy()

    # Load models and state
    nlp = load_spacy(spacy_model)
    logger.info(f"Loading embedding model: {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    state = load_state(state_file)

    # Preprocess all concepts once upfront
    concept_lists, raw_to_pp = preprocess_all_concepts(df, raw_concepts_col, nlp)
    all_unique_pp = set(raw_to_pp.values())

    # Determine which preprocessed concepts belong to which phase
    if target_year is not None and year_col in df.columns:
        logger.info(f"Splitting by target_year={target_year}")
        mask_before = df[year_col] < target_year
        mask_target = df[year_col] >= target_year
        logger.info(
            f"Historical: {mask_before.sum()} rows, target: {mask_target.sum()} rows"
        )

        # Collect preprocessed concepts per phase
        pp_before = set()
        for idx in df.index[mask_before]:
            for raw in concept_lists[idx]:
                pp = raw_to_pp.get(raw)
                if pp:
                    pp_before.add(pp)

        pp_target = set()
        for idx in df.index[mask_target]:
            for raw in concept_lists[idx]:
                pp = raw_to_pp.get(raw)
                if pp:
                    pp_target.add(pp)

        # Phase 1: historical
        if pp_before:
            logger.info(
                f"=== Phase 1: Historical concepts ({len(pp_before)} unique) ==="
            )
            process_new_concepts(
                pp_before, state, encoder, similarity_threshold, batch_size
            )

        # Phase 2: target year
        if pp_target:
            logger.info(
                f"=== Phase 2: Target year concepts ({len(pp_target)} unique) ==="
            )
            process_new_concepts(
                pp_target, state, encoder, similarity_threshold, batch_size
            )
    else:
        # No year splitting — process everything at once
        logger.info(f"Processing all {len(all_unique_pp)} unique preprocessed concepts")
        process_new_concepts(
            all_unique_pp, state, encoder, similarity_threshold, batch_size
        )

    # Build output using the single preprocessed mapping
    logger.info("Building output...")
    normalized, cluster_ids, preprocessed = build_output(
        concept_lists, raw_to_pp, state
    )

    df[normalized_concepts_column] = normalized
    df[cluster_ids_column] = cluster_ids
    if return_details:
        df["preprocessed_concepts"] = preprocessed

    all_canonical = {c for lst in normalized for c in lst if c}
    n_clusters = len(state["canonical_embeddings"])
    logger.info(
        f"Done: {len(set(raw_to_pp.values()))} preprocessed -> {len(all_canonical)} canonical ({n_clusters} clusters)"
    )

    cluster_df = build_cluster_table(state)

    if save:
        save_state(state, state_file)

    return df, cluster_df, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize scientific concepts")
    parser.add_argument("--config", default="configs/concept_normalization.yaml")
    cfg = load_config(parser.parse_args().config)

    cols = cfg.get("columns", {})

    logger.info(f"Reading {cfg['input_file']}")
    df = pd.read_csv(cfg["input_file"])
    logger.info(f"Loaded {len(df)} rows")

    df_out, cluster_df, state = normalize_concepts(
        df,
        raw_concepts_col=cols.get("raw_concepts", "raw_concepts"),
        year_col=cols.get("year_column", "year"),
        target_year=cfg.get("target_year"),
        similarity_threshold=cfg.get("similarity_threshold", 0.92),
        embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        spacy_model=cfg.get("spacy_model", "en_core_web_sm"),
        batch_size=cfg.get("batch_size", 256),
        return_details=cfg.get("return_details", False),
        state_file=cfg.get("state_file"),
        save=True,
        normalized_concepts_column=cols.get(
            "normalized_concepts_column", "normalized_concepts"
        ),
        cluster_ids_column=cols.get("cluster_ids_column", "cluster_ids"),
    )

    output_file = cfg["output_file"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_file, index=False)
    logger.info(f"Saved -> {output_file}")

    if clusters_file := cfg.get("clusters_output_file"):
        Path(clusters_file).parent.mkdir(parents=True, exist_ok=True)
        cluster_df.to_csv(clusters_file, index=False)
        logger.info(f"Clusters -> {clusters_file} ({len(cluster_df)} entries)")
