from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # paths
    pdf_path: str = "data/The_Oxford_3000_by_CEFR_level.pdf"
    csv_path: str = "data/oxford3000_cefr.csv"
    sentences_csv_path: str = "data/oxford3000_with_sentences.csv"
    outputs_dir: str = "outputs"
    pred_dir: str = "outputs/predictions"
    fig_dir: str = "outputs/figures"

    # classes
    all_classes: Tuple[str, ...] = ("A1", "A2", "B1", "B2")
    base_classes: Tuple[str, ...] = ("A1", "A2")
    novel_classes: Tuple[str, ...] = ("B1", "B2")

    # split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # few-shot
    shots: Tuple[int, ...] = (1, 5, 10, 20)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # 【修改】换成精度更高的 mpnet，维度从 384 -> 768
    embedding_mode: str = "sbert"
    embedding_dim: int = 768
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # model search
    knn_candidates: Tuple[int, ...] = (1, 3, 5, 7)
    knn_metrics: Tuple[str, ...] = ("cosine", "euclidean")
    svm_c_candidates: Tuple[float, ...] = (0.1, 1.0, 10.0)

    # optional baseline
    enable_zeroshot: bool = True

    # random
    global_seed: int = 42

CFG = Config()
