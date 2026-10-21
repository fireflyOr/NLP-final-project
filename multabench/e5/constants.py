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

# Frozen PCA-30 columns + out-of-fold text-model predictions (uses y -> compare against Joint TAR)
E5_SMALL_STACK = "e5-small-v2|stack"        # MLP on frozen E5 embeddings, 5-fold cross-fitted
E5_SMALL_TAR_STACK = "e5-small-v2|tarstack" # LoRA-fine-tuned E5 + head (as in TAR), 3-fold cross-fitted
E5_SMALL_TAR_FSTACK = "e5-small-v2|tar-fstack"  # Joint TAR's PCA-30 columns + out-of-fold frozen-MLP text predictions
TAR_STACK_ENCODERS = {E5_SMALL_TAR_FSTACK: ("frozen", 5)}
STACK_ENCODERS = {E5_SMALL_STACK: ("frozen", 5), E5_SMALL_TAR_STACK: ("finetuned", 3)}

# Target-relevant concept filtering before PCA (uses y -> compare against Joint TAR). Value: variant.
E5_SMALL_RELSAE = "e5-small-v2|relsae"            # main: SAE latents selected by conditional relevance, subtractive, PCA-30
E5_SMALL_RELSAE_ADD = "e5-small-v2|relsae-add"    # variant: additive map-back (kept concepts only)
E5_SMALL_RELSAE_UNI = "e5-small-v2|relsae-uni"    # ablation: univariate relevance (ignores the table)
E5_SMALL_RELSAE_RAND = "e5-small-v2|relsae-rand"  # control: random latents, same number as selected
E5_SMALL_RELDIM = "e5-small-v2|reldim"            # control: same selection on raw E5 dimensions (supervised PCA)
E5_SMALL_PLS = "e5-small-v2|pls"                  # control: PLS to 30 components
# v2: remove only clearly irrelevant concepts (score below its own permutation median), larger SAE (512, k=32)
E5_SMALL_RELSAE2 = "e5-small-v2|relsae2"
E5_SMALL_RELSAE2_UNI = "e5-small-v2|relsae2-uni"
E5_SMALL_RELSAE2_RAND = "e5-small-v2|relsae2-rand"
# v3: sibling-aware group-shadow pruning with a 2048-latent SAE, on frozen or TAR-tuned embeddings. Value: (tuned, kind)
E5_SMALL_GSAE = "e5-small-v2|gsae"                # frozen E5 -> SAE -> joint pruning -> PCA-30
E5_SMALL_TAR_GSAE = "e5-small-v2|tar-gsae"        # TAR-tuned E5 -> SAE -> joint pruning -> PCA-30   (main)
E5_SMALL_TAR_GSAE_RAND = "e5-small-v2|tar-gsae-rand"  # control: same number of random concepts removed
E5_SMALL_TAR_GSAE_RANDVAR = "e5-small-v2|tar-gsae-randvar"  # control: random concepts with the same removed variance
E5_SMALL_TAR_GSAE_CV = "e5-small-v2|tar-gsae-cv"    # v4: held-out importance, harm ranking, validated amount
E5_SMALL_TAR_GSAE_XF = "e5-small-v2|tar-gsae-xf"    # v5: removal amount chosen by nested cross-fitting of TAR
PRUNE_ENCODERS = {E5_SMALL_TAR_GSAE_XF: (True, "xfit"), E5_SMALL_TAR_GSAE_CV: (True, "heldout"), E5_SMALL_GSAE: (False, "sae"), E5_SMALL_TAR_GSAE: (True, "sae"), E5_SMALL_TAR_GSAE_RAND: (True, "random"),
                  E5_SMALL_TAR_GSAE_RANDVAR: (True, "random_var")}

RELEVANCE_ENCODERS = {E5_SMALL_RELSAE2: "v2_cond", E5_SMALL_RELSAE2_UNI: "v2_uni", E5_SMALL_RELSAE2_RAND: "v2_rand",E5_SMALL_RELSAE: "sae_cond", E5_SMALL_RELSAE_ADD: "sae_cond_add", E5_SMALL_RELSAE_UNI: "sae_uni", E5_SMALL_RELSAE_RAND: "sae_rand",
                      E5_SMALL_RELDIM: "dim_cond", E5_SMALL_PLS: "pls"}

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
    # tune_encoder=True so the pipeline passes y (and the LoRA args) to the text encoder
    "e5-small-stack": Encoder(encoder_name=E5_SMALL_STACK, tune_encoder=True),
    "e5-small-tar-fstack": Encoder(encoder_name=E5_SMALL_TAR_FSTACK, tune_encoder=True),
    "e5-small-tar-stack": Encoder(encoder_name=E5_SMALL_TAR_STACK, tune_encoder=True),
    "e5-small-relsae": Encoder(encoder_name=E5_SMALL_RELSAE, tune_encoder=True),
    "e5-small-relsae2": Encoder(encoder_name=E5_SMALL_RELSAE2, tune_encoder=True),
    "e5-small-relsae2-uni": Encoder(encoder_name=E5_SMALL_RELSAE2_UNI, tune_encoder=True),
    "e5-small-relsae2-rand": Encoder(encoder_name=E5_SMALL_RELSAE2_RAND, tune_encoder=True),
    "e5-small-relsae-add": Encoder(encoder_name=E5_SMALL_RELSAE_ADD, tune_encoder=True),
    "e5-small-relsae-uni": Encoder(encoder_name=E5_SMALL_RELSAE_UNI, tune_encoder=True),
    "e5-small-relsae-rand": Encoder(encoder_name=E5_SMALL_RELSAE_RAND, tune_encoder=True),
    "e5-small-reldim": Encoder(encoder_name=E5_SMALL_RELDIM, tune_encoder=True),
    "e5-small-pls": Encoder(encoder_name=E5_SMALL_PLS, tune_encoder=True),
    "e5-small-gsae": Encoder(encoder_name=E5_SMALL_GSAE, tune_encoder=True),
    "e5-small-tar-gsae": Encoder(encoder_name=E5_SMALL_TAR_GSAE, tune_encoder=True),
    "e5-small-tar-gsae-xf": Encoder(encoder_name=E5_SMALL_TAR_GSAE_XF, tune_encoder=True),
    "e5-small-tar-gsae-cv": Encoder(encoder_name=E5_SMALL_TAR_GSAE_CV, tune_encoder=True),
    "e5-small-tar-gsae-randvar": Encoder(encoder_name=E5_SMALL_TAR_GSAE_RANDVAR, tune_encoder=True),
    "e5-small-tar-gsae-rand": Encoder(encoder_name=E5_SMALL_TAR_GSAE_RAND, tune_encoder=True),
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
