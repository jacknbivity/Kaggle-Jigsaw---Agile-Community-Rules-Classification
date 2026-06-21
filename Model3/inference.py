import os
os.environ["VLLM_USE_V1"] = "0"

import random
import vllm
import torch
import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm.lora.request import LoRARequest
from utils import build_dataset, build_prompt, build_prompt_with_rag, build_tta_prompts, build_rag_retriever, retrieve_similar_examples, preprocess_text
from constants import *
import multiprocessing as mp
from tqdm.auto import tqdm

# ==================== 配置开关 ====================
USE_TTA = True          # 测试时增强：4 种正/负例组合
USE_RAG = True          # 检索增强：从训练集检索相似样本
RAG_TOP_K = 2           # RAG 每个类别检索数量
TTA_AGGREGATION = "mean"  # TTA 聚合方式: "mean" 或 "max_vote"
MAX_EXAMPLE_LENGTH = 200 # RAG 检索样本最大字符数（防止超出训练时的上下文长度）
# ================================================

# ★ 微调模型兼容性说明 ★
# 1. TTA：完全安全。仅替换示例内容，prompt 格式与训练时完全一致
# 2. RAG：格式兼容。检索样本融入 "Examples:" 块，使用相同 {judge_words} Yes/No 格式
# 3. 上下文长度：训练时 max_model_len=2048，RAG 会自动截断过长示例
# 4. 建议先单独测试 TTA（风险最低），确认有效后再叠加 RAG


def run_inference_raw(llm, tokenizer, prompts):
    """
    执行原始推理，返回 Yes/No 的概率。
    """
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=[positive, negative])],
            logprobs=2,
        ),
        use_tqdm=False,
        lora_request=LoRARequest("lora1", 1, lora_path)
    )
    results = []
    for out in outputs:
        log_probs = {lp.decoded_token: np.exp(lp.logprob) for lp in out.outputs[0].logprobs[0].values()}
        yes_prob = log_probs.get(positive, 0.0)
        no_prob = log_probs.get(negative, 0.0)
        results.append({"Yes": yes_prob, "No": no_prob})
    return results


def aggregate_tta_results(tta_results, method="mean"):
    """
    聚合 TTA 的 4 组预测结果。
    
    参数:
        tta_results: list of list, shape [num_samples, 4, 2]（每样本 4 组增强 × Yes/No）
        method: "mean" 平均概率, "max_vote" 多数投票
    """
    tta_arr = np.array(tta_results)  # [num_samples, 4]
    
    if method == "mean":
        return tta_arr.mean(axis=1)
    elif method == "max_vote":
        # 多数投票：每组选 max(Yes,No) 的类别，统计 Yes 的比例
        votes = (tta_arr > 0.5).astype(float)
        return votes.mean(axis=1)
    else:
        return tta_arr.mean(axis=1)


def run_inference_on_device(df_slice, retriever=None):
    """
    在单张 GPU 上执行 RAG + TTA 推理。
    
    ★ 微调模型安全策略：
    - TTA: prompt 格式与训练时 100% 一致（仅交换示例内容）
    - RAG: 检索样本融入原有 Examples 块，格式保持 {judge_words} Yes/No
    - 检索示例自动截断到 MAX_EXAMPLE_LENGTH，防止超出训练上下文分布
    - max_model_len 保持与训练一致（2048），长 prompt 由 tokenizer truncation 处理
    """
    llm = vllm.LLM(
        base_model_path,
        quantization="gptq" if use_gptq else None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype=torch.float16,
        enforce_eager=True,
        max_model_len=2048,  # ★ 保持与训练时一致，不随意扩大
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()
    
    all_yes_probs = []
    all_row_ids = []
    
    for idx in tqdm(range(len(df_slice)), desc="Processing samples"):
        row = df_slice.iloc[idx]
        row_id = row["row_id"]
        
        if USE_TTA:
            # === TTA 模式：生成 4 组增强 prompt ===
            tta_prompts = build_tta_prompts(row, retriever=retriever, use_rag=USE_RAG, rag_top_k=RAG_TOP_K)
            
            # 批量推理 4 组（vLLM prefix caching 自动加速）
            tta_results = run_inference_raw(llm, tokenizer, tta_prompts)
            
            # 聚合 4 组结果
            yes_probs = [r["Yes"] for r in tta_results]
            agg_yes = aggregate_tta_results([yes_probs], method=TTA_AGGREGATION)[0]
            
            all_yes_probs.append(agg_yes)
            all_row_ids.append(row_id)
        else:
            # === 单次推理模式 ===
            if USE_RAG and retriever is not None:
                retrieved_pos, retrieved_neg = retrieve_similar_examples(
                    row, retriever, top_k=RAG_TOP_K
                )
                # 截断过长示例以保持上下文长度在训练分布内
                retrieved_pos = [t[:MAX_EXAMPLE_LENGTH] for t in retrieved_pos]
                retrieved_neg = [t[:MAX_EXAMPLE_LENGTH] for t in retrieved_neg]
                row_copy = row.copy()
                row_copy["positive_example"] = row["positive_example_1"]
                row_copy["negative_example"] = row["negative_example_1"]
                prompt = build_prompt_with_rag(row_copy, retrieved_pos, retrieved_neg)
            else:
                row_copy = row.copy()
                row_copy["positive_example"] = random.choice([row["positive_example_1"], row["positive_example_2"]])
                row_copy["negative_example"] = random.choice([row["negative_example_1"], row["negative_example_2"]])
                prompt = build_prompt(row_copy)
            
            results = run_inference_raw(llm, tokenizer, [prompt])
            all_yes_probs.append(results[0]["Yes"])
            all_row_ids.append(row_id)
    
    predictions = pd.DataFrame({
        "row_id": all_row_ids,
        positive: all_yes_probs,
    })
    predictions[negative] = 1.0 - predictions[positive]
    return predictions


def worker(device_id, df_slice, return_dict, retriever=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")
    print(f"[Worker {device_id}] Config: TTA={USE_TTA}, RAG={USE_RAG}, RAG_TOP_K={RAG_TOP_K}")
    preds = run_inference_on_device(df_slice, retriever=retriever)
    return_dict[device_id] = preds


def main():
    print("=" * 50)
    print(f"RAG+TTA Inference Configuration:")
    print(f"  USE_TTA = {USE_TTA}")
    print(f"  USE_RAG = {USE_RAG}")
    print(f"  RAG_TOP_K = {RAG_TOP_K}")
    print(f"  TTA_AGGREGATION = {TTA_AGGREGATION}")
    print("=" * 50)
    
    # 加载测试数据
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    
    # 预处理 body 文本
    test_df["body"] = test_df["body"].apply(preprocess_text)
    for col in ["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]:
        test_df[col] = test_df[col].apply(preprocess_text)
    
    # 构建 RAG 检索器（如果需要）
    retriever = None
    if USE_RAG:
        print("\nBuilding RAG retriever from training data...")
        retriever = build_rag_retriever()
        print("RAG retriever ready.\n")
    
    # 数据分片（双 GPU）
    mid = len(test_df) // 2
    df0 = test_df.iloc[:mid].reset_index(drop=True)
    df1 = test_df.iloc[mid:].reset_index(drop=True)

    manager = mp.Manager()
    return_dict = manager.dict()
    p0 = mp.Process(target=worker, args=(0, df0, return_dict, retriever))
    p1 = mp.Process(target=worker, args=(1, df1, return_dict, retriever))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    predictions = pd.concat([return_dict[0], return_dict[1]], ignore_index=True)
    submission = predictions[["row_id", positive]].rename(columns={positive: "rule_violation"})
    
    output_path = "/kaggle/working/Qwen3_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Predictions: {len(submission)} rows, mean={submission['rule_violation'].mean():.4f}")

if __name__ == "__main__":
    main()