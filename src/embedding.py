import re
import numpy as np
from sklearn.preprocessing import normalize
from .config import CFG
import nltk
from nltk.corpus import wordnet

try:
    wordnet.synsets('apple')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    from wordfreq import zipf_frequency
except ImportError:
    def zipf_frequency(w, lang):
        return 4.0

def count_syllables(word):
    word = str(word).lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def calc_complexity(word):
    word = str(word).lower()
    if len(word) == 0:
        return 0.0
    return len(set(word)) / len(word)

def get_wordnet_depth(word):
    synsets = wordnet.synsets(str(word))
    if not synsets:
        return 0.0
    return float(max([s.min_depth() for s in synsets]))

# 【新增】AoA 近似估计：用词频 + 词长 + WordNet深度的线性组合模拟习得年龄
# 学术依据：AoA 与 Zipf频率强负相关(r≈-0.7)，与词长正相关(r≈0.4)
# 真实 AoA 数据集(Kuperman et al. 2012)需要额外文件，这里用代理特征近似
def estimate_aoa(word):
    freq = zipf_frequency(str(word), 'en')
    length = len(str(word))
    depth = get_wordnet_depth(word)
    # 频率越低、词越长、语义越深 → AoA 越大（越晚习得）
    aoa_proxy = -0.6 * freq + 0.3 * length + 0.1 * depth
    return float(aoa_proxy)

# 【新增】词缀特征：识别拉丁/希腊语源词缀（通常是 B1/B2 级别的标志）
# 学术依据：Anglo-Saxon 词根词汇偏 A1/A2，Latinate 词汇偏 B1/B2
LATINATE_PREFIXES = ('ab','ac','ad','al','ap','as','com','con','de','dis',
                     'ex','hyp','im','in','inter','mis','ob','per','pre',
                     'pro','re','sub','super','trans','un','under')
LATINATE_SUFFIXES = ('tion','sion','ity','ance','ence','ment','ous','ive',
                     'ize','ise','ify','ate','ary','ory','al','ic','ical')

def latinate_score(word):
    w = str(word).lower()
    score = 0.0
    for p in LATINATE_PREFIXES:
        if w.startswith(p) and len(w) > len(p) + 2:
            score += 1.0
            break
    for s in LATINATE_SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            score += 1.0
            break
    return score  # 0, 1, 或 2

class Embedder:
    def __init__(self, mode=CFG.embedding_mode, dim=CFG.embedding_dim):
        self.mode = mode
        self.dim = dim
        self.model = None

        if self.mode == "sbert":
            print(f"[INFO] Loading SentenceTransformer: {CFG.sbert_model_name}...")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(CFG.sbert_model_name)
                try:
                    self.dim = self.model.get_sentence_embedding_dimension()
                except AttributeError:
                    self.dim = self.model.get_embedding_dimension()
                print(f"[INFO] SBERT loaded! embedding_dim={self.dim}")
            except Exception as e:
                print(f"[ERROR] Failed to load SBERT: {e}")
                self.mode = "hash"

    def _hash_vec(self, w: str):
        rs = np.random.RandomState(abs(hash(w)) % (2 ** 32))
        return rs.uniform(-1e-4, 1e-4, self.dim).astype(np.float32)

    def transform(self, words, sentences=None, use_context=True):
        # ── SBERT 语义向量 ──────────────────────────────────
        X_text = np.zeros((len(words), self.dim), dtype=np.float32)
        if self.mode == "sbert" and self.model is not None:
            if sentences is not None and use_context:
                texts_to_encode = [
                    f'Target Word: "{w}". Context: {s}'
                    for w, s in zip(words, sentences)
                ]
            else:
                texts_to_encode = [str(w) for w in words]
            X_text = self.model.encode(texts_to_encode, show_progress_bar=False)
        else:
            for i, w in enumerate(words):
                X_text[i] = self._hash_vec(str(w))

        # ── 标量特征（全部基于目标词本身，与例句无关）────────
        lengths = np.array([len(str(w)) for w in words],
                           dtype=np.float32).reshape(-1, 1)
        lengths_scaled = (lengths - 7.0) / 3.0

        freqs = np.array([zipf_frequency(str(w), 'en') for w in words],
                         dtype=np.float32).reshape(-1, 1)
        freqs_scaled = (freqs - 4.0) / 2.0

        syllables = np.array([count_syllables(w) for w in words],
                             dtype=np.float32).reshape(-1, 1)
        syllables_scaled = (syllables - 2.0) / 1.5

        complexities = np.array([calc_complexity(w) for w in words],
                                dtype=np.float32).reshape(-1, 1)
        complexities_scaled = (complexities - 0.7) / 0.2

        polysemy = np.array([len(wordnet.synsets(str(w))) for w in words],
                            dtype=np.float32).reshape(-1, 1)
        polysemy_scaled = (polysemy - 5.0) / 5.0

        depths = np.array([get_wordnet_depth(w) for w in words],
                          dtype=np.float32).reshape(-1, 1)
        depths_scaled = (depths - 6.0) / 3.0

        # 【新增】AoA 代理特征（最强 CEFR 信号）
        aoa = np.array([estimate_aoa(w) for w in words],
                       dtype=np.float32).reshape(-1, 1)
        aoa_scaled = (aoa - (-1.5)) / 1.5   # 中心化，均值约-1.5，std约1.5

        # 【新增】词缀拉丁化程度（B1/B2 判别特征）
        latinate = np.array([latinate_score(w) for w in words],
                            dtype=np.float32).reshape(-1, 1)
        latinate_scaled = (latinate - 0.8) / 0.6

        # ── 拼接，768维下标量权重 ×2 ─────────────────────────
        X_combined = np.hstack([
            X_text            * 1.0,
            lengths_scaled    * 0.4,
            freqs_scaled      * 0.8,   # 词频：最强信号
            syllables_scaled  * 0.4,
            complexities_scaled * 0.3,
            polysemy_scaled   * 0.5,
            depths_scaled     * 0.5,
            aoa_scaled        * 0.9,   # 【新增】AoA：第二强信号，权重仅次于词频
            latinate_scaled   * 0.6,   # 【新增】词缀：B1/B2 判别器
        ])

        X_combined = normalize(X_combined, norm='l2')
        return X_combined
