import os
import re
import argparse
import pdfplumber
import pandas as pd
from .config import CFG

POS_MARKERS = {
    "n", "v", "adj", "adv", "prep", "conj", "pron", "det",
    "exclam", "auxiliary", "modal", "number", "article"
}

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.pred_dir, exist_ok=True)
    os.makedirs(CFG.fig_dir, exist_ok=True)

def clean_token(token: str) -> str:
    t = token.strip().lower().replace("’", "'")
    t = re.sub(r"[^a-z'\-]", "", t)
    return t

def parse_pdf_to_rows(pdf_path: str):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            all_text.append(txt)

    text = "\n".join(all_text)

    # OCR/抽取常见混淆修正
    text = text.replace("Bl", "B1").replace("B I", "B1").replace("A I", "A1")
    text = re.sub(r"\r", "\n", text)

    # 按 A1/A2/B1/B2 四大区块切分
    pattern = r"\bA1\b(.*?)\bA2\b(.*?)\bB1\b(.*?)\bB2\b(.*)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        raise ValueError("无法定位 A1/A2/B1/B2 四个区块，请检查PDF文本抽取结果。")

    blocks = {
        "A1": m.group(1),
        "A2": m.group(2),
        "B1": m.group(3),
        "B2": m.group(4),
    }

    rows = []
    for level, block in blocks.items():
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z'\-]*\b", block)
        for t in tokens:
            w = clean_token(t)
            if not w:
                continue
            if w in POS_MARKERS:
                continue
            if len(w) <= 1 and w != "a":
                continue
            if re.fullmatch(r"\d+", w):
                continue
            rows.append((w, level))

    return rows

def resolve_word_label_conflict(df: pd.DataFrame) -> pd.DataFrame:
    # 同一个词若出现多个标签，按计数投票取最多
    if not df.duplicated(subset=["word"], keep=False).any():
        return df

    voted = (
        df.groupby(["word", "label"]).size().reset_index(name="cnt")
        .sort_values(["word", "cnt"], ascending=[True, False])
        .drop_duplicates(subset=["word"], keep="first")
    )
    return voted[["word", "label"]]

def build_csv_from_pdf(pdf_path: str = CFG.pdf_path, csv_path: str = CFG.csv_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rows = parse_pdf_to_rows(pdf_path)
    df = pd.DataFrame(rows, columns=["word", "label"])
    df = df[df["label"].isin(CFG.all_classes)].copy()
    df["word"] = df["word"].str.lower().str.strip()
    df = df[df["word"].str.len() > 0]
    df = df.drop_duplicates(subset=["word", "label"])
    df = resolve_word_label_conflict(df)
    df = df.sort_values(["label", "word"]).reset_index(drop=True)

    # 关键校验：必须四类齐全
    counts = df["label"].value_counts()
    missing = [c for c in CFG.all_classes if c not in counts.index]
    if missing:
        raise ValueError(f"CSV缺少类别: {missing}，当前分布:\n{counts}")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] CSV saved to: {csv_path}, n={len(df)}")
    print(df["label"].value_counts())

def load_dataset(csv_path: str = CFG.sentences_csv_path) -> pd.DataFrame: # 【修改1】默认路径改为 sentences_csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. 请先运行 generate_sentences.py 生成带例句的数据。")
    
    df = pd.read_csv(csv_path)
    
    # 【修改2】增加对 sentence 列的校验
    if not {"word", "label", "sentence"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: word, label, sentence")
        
    # 【修改3】清理空值时把 sentence 也加上
    df = df.dropna(subset=["word", "label", "sentence"]).copy()
    
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["sentence"] = df["sentence"].astype(str).str.strip() # 【修改4】清理例句文本
    df = df[df["label"].isin(CFG.all_classes)].reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_csv", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.build_csv:
        build_csv_from_pdf()
    else:
        print("Use: python -m src.io_data --build_csv")
