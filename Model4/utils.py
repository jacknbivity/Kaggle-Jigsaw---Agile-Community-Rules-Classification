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