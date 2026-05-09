import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CFG


def split_train_val_test(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df, test_size=(1 - CFG.train_ratio),
        stratify=df["label"], random_state=seed
    )
    rel_test = CFG.test_ratio / (CFG.val_ratio + CFG.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_base_novel(train_df: pd.DataFrame):
    base_df = train_df[train_df["label"].isin(CFG.base_classes)].reset_index(drop=True)
    novel_df = train_df[train_df["label"].isin(CFG.novel_classes)].reset_index(drop=True)
    return base_df, novel_df


def sample_support(base_df: pd.DataFrame, novel_df: pd.DataFrame, k: int, seed: int):
    rng = np.random.RandomState(seed)
    parts = []

    for c in CFG.base_classes:
        d = base_df[base_df["label"] == c]
        parts.append(d.sample(n=k, random_state=rng))

    for c in CFG.novel_classes:
        d = novel_df[novel_df["label"] == c]
        parts.append(d.sample(n=k, random_state=rng))

    support_df = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    return support_df
