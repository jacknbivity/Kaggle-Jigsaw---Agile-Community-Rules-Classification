import numpy as np
import pandas as pd
import re
from datasets import Dataset
from constants import *

# 缩略词映射字典
CONTRACTION_MAPPING = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would", "there'd've": "there would have",
    "there's": "there is", "here's": "here is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
    "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

def clean_punctuation(text):
    """清理无意义的标点符号,保留有意义的标点"""
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s!,.?:;\'"\-@#*/]<>', '', text)
    
    return text

def expand_contractions(text):
    """扩展文本中的缩略词并清理标点"""
    if not isinstance(text, str):
        return text
        
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in CONTRACTION_MAPPING.keys()) + r')\b')
    
    def replace(match):
        return CONTRACTION_MAPPING.get(match.group(0).lower(), match.group(0))
    
    text = pattern.sub(replace, text)
    text = clean_punctuation(text)
    
    return text

def preprocess_text(text):
    """综合文本预处理函数"""
    if not isinstance(text, str):
        return ""
    
    # 扩展缩略词和清理标点
    text = expand_contractions(text)
    
    return text

def build_prompt(row):
    return f"""{system_prompt}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{judge_words} Yes
2) {row["negative_example"]}
{judge_words} No
Comment: {row["body"]}
{judge_words}"""

def get_df():
    merge = list()
    if use_train:
        train_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
        train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                                "positive_example_1", "positive_example_2", 
                                "negative_example_1", "negative_example_2"]].copy()
        train_df["positive_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["positive_example_1"], train_df["positive_example_2"])
        train_df["negative_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["negative_example_1"], train_df["negative_example_2"])
        train_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
        merge.append(train_df)
    test_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_dataset = test_dataset.groupby('rule', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"Select {len(test_dataset)} test data")
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule", "subreddit", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]].copy()
            body_col = f"{violation_type}_example_{i}"
            other_positive_col = f"{violation_type}_example_{3-i}"
            sub_dataset["body"] = sub_dataset[body_col]
            sub_dataset[f"{violation_type}_example"] = sub_dataset[other_positive_col]
            anti_violation_type = "negative" if violation_type == "positive" else "positive"
            sub_dataset[f"{anti_violation_type}_example"] = np.where(np.random.rand(len(sub_dataset)) < 0.5, sub_dataset[f"{anti_violation_type}_example_1"], sub_dataset[f"{anti_violation_type}_example_2"])
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
            merge.append(sub_dataset)
    return pd.concat(merge, axis=0).drop_duplicates(ignore_index=True)

def build_dataset(df):
    df["prompt"] = df.apply(build_prompt, axis=1)
    columns = ["prompt"]
    if "rule_violation" in df:
        df["completion"] = df["rule_violation"].map({
            1: positive,
            0: negative,})
        columns.append("completion")
    dataset = Dataset.from_pandas(df[columns])
    return dataset


# ==================== RAG 检索增强模块 ====================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def build_rag_retriever(train_csv_path="/kaggle/input/jigsaw-agile-community-rules/train.csv"):
    """
    构建 TF-IDF 检索器，按规则分组建立索引。
    返回:
        retriever: dict, key=rule, value={"vectorizer": TfidfVectorizer, "matrix": csr_matrix, "df": DataFrame}
    """
    train_df = pd.read_csv(train_csv_path)
    # 预处理 body 文本
    train_df["body_clean"] = train_df["body"].apply(preprocess_text)
    
    retriever = {}
    for rule, group in train_df.groupby("rule"):
        bodies = group["body_clean"].tolist()
        if len(bodies) < 2:
            continue
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(bodies)
        retriever[rule] = {
            "vectorizer": vectorizer,
            "matrix": tfidf_matrix,
            "df": group.reset_index(drop=True),
        }
    
    print(f"RAG retriever built: {len(retriever)} rules indexed")
    return retriever


def retrieve_similar_examples(row, retriever, top_k=2):
    """
    为单条测试样本检索同规则下最相似的训练样本。
    
    参数:
        row: 测试样本 (包含 rule, body)
        retriever: build_rag_retriever 返回的检索器
        top_k: 每个类别（正/负例）检索的数量
    
    返回:
        retrieved_pos: list of body strings (违规样本)
        retrieved_neg: list of body strings (不违规样本)
    """
    rule = row["rule"]
    body = preprocess_text(row["body"]) if isinstance(row["body"], str) else ""
    
    if rule not in retriever or not body.strip():
        return [], []
    
    rule_data = retriever[rule]
    vectorizer = rule_data["vectorizer"]
    tfidf_matrix = rule_data["matrix"]
    df = rule_data["df"]
    
    # 将查询文本向量化
    query_vec = vectorizer.transform([body])
    
    # 计算与所有训练样本的余弦相似度
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 按相似度排序
    top_indices = similarities.argsort()[::-1]
    
    retrieved_pos = []
    retrieved_neg = []
    
    for idx in top_indices:
        if len(retrieved_pos) >= top_k and len(retrieved_neg) >= top_k:
            break
        candidate = df.iloc[idx]
        # 跳过与查询完全相同的文本
        if candidate["body_clean"] == body:
            continue
        if candidate["rule_violation"] == 1 and len(retrieved_pos) < top_k:
            retrieved_pos.append(candidate["body"])
        elif candidate["rule_violation"] == 0 and len(retrieved_neg) < top_k:
            retrieved_neg.append(candidate["body"])
    
    return retrieved_pos, retrieved_neg


def build_prompt_with_rag(row, retrieved_pos=None, retrieved_neg=None):
    """
    构建带 RAG 检索增强的 prompt。
    
    ★ 关键设计：保持与训练时完全一致的 prompt 格式 ★
    - 检索样本直接融入 "Examples:" 块，不引入新标签头
    - 使用与训练相同的 {judge_words} Yes/No 格式
    - 编号连续：检索样本在前，给定示例在后
    """
    examples = []
    example_num = 1
    
    # 检索到的相似样本（真实训练数据，带标签）
    for body in (retrieved_pos or []):
        examples.append(f"{example_num}) {body}\n{judge_words} Yes")
        example_num += 1
    for body in (retrieved_neg or []):
        examples.append(f"{example_num}) {body}\n{judge_words} No")
        example_num += 1
    
    # 给定的正/负例（来自测试集本身）
    examples.append(f"{example_num}) {row['positive_example']}\n{judge_words} Yes")
    example_num += 1
    examples.append(f"{example_num}) {row['negative_example']}\n{judge_words} No")
    
    examples_str = "\n".join(examples)
    
    # 格式与训练时 build_prompt 完全一致，只是 Example 数量可变
    return f"""{system_prompt}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
{examples_str}
Comment: {row["body"]}
{judge_words}"""


def build_tta_prompts(row, retriever=None, use_rag=False, rag_top_k=2):
    """
    TTA（测试时增强）：利用 positive_example_1/2 和 negative_example_1/2 的 4 种组合。
    可选叠加 RAG 检索增强（格式兼容微调模型）。
    
    返回:
        prompts: list of str, 4 个增强后的 prompt
    """
    prompts = []
    
    # 如果启用 RAG，先检索（4 组 TTA 共享同一批检索结果）
    retrieved_pos, retrieved_neg = [], []
    if use_rag and retriever is not None:
        retrieved_pos, retrieved_neg = retrieve_similar_examples(row, retriever, top_k=rag_top_k)
    
    # 4 种正/负例组合
    for pos_col in ["positive_example_1", "positive_example_2"]:
        for neg_col in ["negative_example_1", "negative_example_2"]:
            aug_row = row.copy()
            aug_row["positive_example"] = row[pos_col]
            aug_row["negative_example"] = row[neg_col]
            
            if use_rag and (retrieved_pos or retrieved_neg):
                prompt = build_prompt_with_rag(aug_row, retrieved_pos, retrieved_neg)
            else:
                prompt = build_prompt(aug_row)
            prompts.append(prompt)
    
    return prompts