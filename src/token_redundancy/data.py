"""MRPC loading and tokenization (plan Section 3).

Splits:
  train -> 90% of GLUE train (fine-tuning, head importance, fixing signal directions, tuning epsilon)
  val   -> 10% of GLUE train (hyperparameters, early stopping)
  dev   -> GLUE validation, 408 pairs (all reported results; never used for tuning)
"""

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from token_redundancy.config import MAX_LENGTH, MODEL_NAME

SPLIT_SEED = 42  # fixed across model seeds so every model sees the same train/val split


def load_mrpc() -> DatasetDict:
    raw = load_dataset("nyu-mll/glue", "mrpc")
    split = raw["train"].train_test_split(test_size=0.1, seed=SPLIT_SEED, stratify_by_column="label")
    return DatasetDict(train=split["train"], val=split["test"], dev=raw["validation"])


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(ds: DatasetDict, tokenizer) -> DatasetDict:
    def encode(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True, max_length=MAX_LENGTH)

    return ds.map(encode, batched=True, remove_columns=["sentence1", "sentence2"])
