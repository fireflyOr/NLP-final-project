"""Constants for E5 text encoders (BERT-based)."""

from multabench.utils.encoders import Encoder

E5_SMALL_V2 = "intfloat/e5-small-v2"
E5_LARGE_V2 = "intfloat/e5-large-v2"

TF_IDF = "tf-idf"  # skrub StringEncoder (TF-IDF + TruncatedSVD), CPU-only

# Frozen E5-small followed by a column transform other than PCA(30). Value: (SAE checkpoint or None, binary).
E5_SMALL_FULL = "e5-small-v2|full"          # control: all 384 dims, no PCA
E5_SMALL_SAE = "e5-small-v2|sae"            # trained TopK SAE, continuous codes
E5_SMALL_SAE_BIN = "e5-small-v2|sae-bin"    # trained TopK SAE, on/off codes
E5_SMALL_RSAE_BIN = "e5-small-v2|rsae-bin"  # control: untrained (random) TopK SAE, on/off codes
E5_SMALL_FSAE = "e5-small-v2|fsae"          # small TopK SAE (30 latents) trained per fold, continuous codes
E5_SMALL_FSAE_BIN = "e5-small-v2|fsae-bin"  # same, on/off codes
E5_SMALL_FSAE50 = "e5-small-v2|fsae50"      # small TopK SAE with 50 latents, continuous codes
E5_SMALL_FSAE50_BIN = "e5-small-v2|fsae50-bin"
E5_SMALL_PCA50 = "e5-small-v2|pca50"        # control for the 50-latent SAE: PCA with 50 components
FOLD_SAE = "fold"                           # marker: train the SAE on the fold instead of loading a checkpoint
PCA = "pca"                                 # marker: PCA with a custom number of components
# Value: (checkpoint name | FOLD_SAE | PCA | None for identity, binary codes, size for FOLD_SAE / PCA)
E5_TRANSFORMS = {
    E5_SMALL_FSAE: (FOLD_SAE, False, 30),
    E5_SMALL_FSAE_BIN: (FOLD_SAE, True, 30),
    E5_SMALL_FSAE50: (FOLD_SAE, False, 50),
    E5_SMALL_FSAE50_BIN: (FOLD_SAE, True, 50),
    E5_SMALL_PCA50: (PCA, False, 50),
    E5_SMALL_FULL: (None, False, None),
    E5_SMALL_SAE: ("e5-small-topk", False, None),
    E5_SMALL_SAE_BIN: ("e5-small-topk", True, None),
    E5_SMALL_RSAE_BIN: ("e5-small-random", True, None),
}

TEXT_ENCODERS = {
    "tfidf": Encoder(encoder_name=TF_IDF, tune_encoder=False),
    "e5-small": Encoder(encoder_name=E5_SMALL_V2, tune_encoder=False),
    "e5-small-tar": Encoder(encoder_name=E5_SMALL_V2, tune_encoder=True),
    "e5-large": Encoder(encoder_name=E5_LARGE_V2, tune_encoder=False),
    "e5-small-full": Encoder(encoder_name=E5_SMALL_FULL, tune_encoder=False),
    "e5-small-sae": Encoder(encoder_name=E5_SMALL_SAE, tune_encoder=False),
    "e5-small-sae-bin": Encoder(encoder_name=E5_SMALL_SAE_BIN, tune_encoder=False),
    "e5-small-rsae-bin": Encoder(encoder_name=E5_SMALL_RSAE_BIN, tune_encoder=False),
    "e5-small-fsae": Encoder(encoder_name=E5_SMALL_FSAE, tune_encoder=False),
    "e5-small-fsae-bin": Encoder(encoder_name=E5_SMALL_FSAE_BIN, tune_encoder=False),
    "e5-small-fsae50": Encoder(encoder_name=E5_SMALL_FSAE50, tune_encoder=False),
    "e5-small-fsae50-bin": Encoder(encoder_name=E5_SMALL_FSAE50_BIN, tune_encoder=False),
    "e5-small-pca50": Encoder(encoder_name=E5_SMALL_PCA50, tune_encoder=False),
}

D_E5_SMALL = 384
D_E5_LARGE = 1024

E5_SMALL_LAYERS = 12
E5_LARGE_LAYERS = 24

E5_DIM = {
    E5_SMALL_V2: D_E5_SMALL,
    E5_LARGE_V2: D_E5_LARGE,
}

E5_NUM_LAYERS = {
    E5_SMALL_V2: E5_SMALL_LAYERS,
    E5_LARGE_V2: E5_LARGE_LAYERS,
}

# BERT layer structure: encoder.layer.{i}.attention.self.{query,key,value}, attention.output.dense, intermediate.dense, output.dense
LORA_TEXT_TARGET_MODULES = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]

# E5 expects a task prefix for embedding; use "passage: " for generic text (e.g. when tuning on a text feature).
# When column name is provided, format is "passage: col_name: col_val".
E5_PASSAGE_PREFIX = "passage: "


def format_e5_passage(col_name: str, col_val: str) -> str:
    """Format text for E5 embedding: passage: col_name: col_val."""
    return f"{E5_PASSAGE_PREFIX}{col_name}: {col_val}"
