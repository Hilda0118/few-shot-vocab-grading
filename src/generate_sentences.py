import pandas as pd
import requests
import time
import concurrent.futures
from tqdm import tqdm

API_KEY = "yourownkey"
API_URL = "https://api.deepseek.com/chat/completions"
MAX_WORKERS = 10

# 【核心】每个 CEFR 级别的语言特征描述，让 DeepSeek 生成带判别信号的例句
CEFR_STYLE_GUIDE = {
    "A1": (
        "CEFR A1 (Beginner). "
        "Use very simple present tense. Short subject-verb-object structure. "
        "Only the most basic everyday vocabulary (body, family, food, numbers). "
        "Example style: 'I have a cat. She is my friend.'"
    ),
    "A2": (
        "CEFR A2 (Elementary). "
        "Use simple past or future tense. Slightly longer sentences with 'because' or 'and'. "
        "Everyday topics like shopping, travel, routines. "
        "Example style: 'She went to the market because she needed food.'"
    ),
    "B1": (
        "CEFR B1 (Intermediate). "
        "Use compound or complex sentences. Include opinions or explanations. "
        "Topics like work, environment, personal experiences. "
        "Example style: 'Although the task was difficult, he managed to complete it efficiently.'"
    ),
    "B2": (
        "CEFR B2 (Upper-Intermediate). "
        "Use subordinate clauses, passive voice, or academic collocations. "
        "Abstract or professional topics. Precise, formal register. "
        "Example style: 'The hypothesis was subsequently validated through a series of controlled experiments.'"
    ),
}

def get_deepseek_sentence(word, label, retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    style = CEFR_STYLE_GUIDE[label]

    # 【修改】Prompt 明确要求例句体现该级别的句式和语域特征
    # 这样 SBERT 编码后，不同级别的句子向量会有更大的判别距离
    prompt = (
        f"You are an expert English linguist specializing in CEFR language assessment.\n"
        f"Write exactly ONE example sentence for the word '{word}'.\n\n"
        f"Target level: {style}\n\n"
        f"CRITICAL RULES:\n"
        f"1. The exact word '{word}' MUST appear in the sentence.\n"
        f"2. The sentence grammar, vocabulary, and register MUST match the level description above.\n"
        f"3. Return ONLY the raw sentence. No quotes, no labels, no explanations."
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    for _ in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                sentence = response.json()['choices'][0]['message']['content'].strip()
                sentence = sentence.strip('"').strip("'")
                if word.lower() in sentence.lower():
                    return sentence
        except Exception:
            time.sleep(1)

    return f'The word {word} is commonly used in everyday language.'

def process_row(row):
    word  = row['word']
    label = row['label']
    sentence = get_deepseek_sentence(word, label)
    return word, label, sentence

if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data/oxford3000_cefr.csv")

    print(f"[INFO] 正在为 {len(df)} 个单词生成带 CEFR 风格特征的例句...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Generating"):
            results.append(future.result())

    new_df = pd.DataFrame(results, columns=['word', 'label', 'sentence'])

    word_order = {word: i for i, word in enumerate(df['word'])}
    new_df['order'] = new_df['word'].map(word_order)
    new_df = new_df.sort_values('order').drop('order', axis=1)

    output_path = "data/oxford3000_with_sentences.csv"
    new_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] 完成！已保存至 {output_path}")
