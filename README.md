# Few-Shot Vocabulary Level Classification

基于小样本学习的英语词汇 CEFR 等级分类框架（A1/A2/B1/B2），对比了 KMeans、KNN、LogReg、GMM、Anchor、Ensemble 等多种方法。

## 环境安装

pip install numpy pandas scikit-learn matplotlib fasttext

## 运行方式

python -m src.generate_sentences
python -m src.run_all

## 词向量下载

本项目使用 FastText 英语预训练词向量 cc.en.300.bin（约7GB），放入 embeddings/ 目录。
下载地址：https://fasttext.cc/docs/en/crawl-vectors.html
