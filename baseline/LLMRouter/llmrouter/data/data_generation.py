#!/usr/bin/env python3
"""
Data Generation Script - Step 2a: Generate Query Data JSONL Files

This script generates query data JSONL files (train/test) from 11 diverse benchmark datasets.
The output format matches StandardQueryData format without embeddings.

Input: Config YAML file (optional, can use command-line args)
Output: query_data_train.jsonl and query_data_test.jsonl files

Usage:
    python data_generation.py --config config.yaml
    python data_generation.py --sample N --output_train PATH --output_test PATH

Examples:
    python data_generation.py --config llmrouter/data/sample_config.yaml
    python data_generation.py --sample 100 --output_train data/query_train.jsonl --output_test data/query_test.jsonl
    python data_generation.py --sample 10 --test  # Quick test with 10 samples
"""

import os
import time
import random
import json
import argparse
import yaml

from datasets import load_dataset

# Import utils
from llmrouter.utils import setup_environment, CASE_NUM
from llmrouter.data.data_loader import DataLoader

# Setup environment
setup_environment()


def get_n_samples(N=10, random_seed=42, cache_dir=None):
    """Extract samples from all datasets

    Args:
        N: Number of samples per dataset
        random_seed: Random seed for reproducibility
        cache_dir: Optional cache directory for datasets. If None, uses default HuggingFace cache.
    """
    random.seed(random_seed)

    # Initialize empty lists for each dataset
    natural_qa_samples = []
    trivia_qa_samples = []
    mmlu_samples = []
    gpqa_samples = []
    mbpp_samples = []
    humaneval_samples = []
    gsm8k_samples = []
    commonsense_qa_samples = []
    math_samples = []
    openbook_qa_samples = []
    arc_challenge_samples = []

    # 1. Natural QA dataset
    try:
        natural_qa = load_dataset(
            "RUC-NLPIR/FlashRAG_datasets", "nq", cache_dir=cache_dir
        )
        split_name = "train" if "train" in natural_qa else list(natural_qa.keys())[0]
        dataset_size = len(natural_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            natural_qa_samples = [natural_qa[split_name][i] for i in indices]
        else:
            natural_qa_samples = [
                natural_qa[split_name][i] for i in range(dataset_size)
            ]
        print(
            f"Successfully extracted {len(natural_qa_samples)} samples from Natural QA"
        )
    except Exception as e:
        print(f"Error extracting from Natural QA: {e}")

    # 2. Trivia QA dataset
    try:
        trivia_qa = load_dataset("trivia_qa", "rc.nocontext", cache_dir=cache_dir)
        split_name = "train" if "train" in trivia_qa else list(trivia_qa.keys())[0]
        dataset_size = len(trivia_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            trivia_qa_samples = [trivia_qa[split_name][i] for i in indices]
        else:
            trivia_qa_samples = [trivia_qa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(trivia_qa_samples)} samples from Trivia QA")
    except Exception as e:
        print(f"Error extracting from Trivia QA: {e}")

    # 3. MMLU dataset
    try:
        mmlu = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
        split_name = (
            "auxiliary_train" if "auxiliary_train" in mmlu else list(mmlu.keys())[0]
        )
        dataset_size = len(mmlu[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            mmlu_samples = [mmlu[split_name][i] for i in indices]
        else:
            mmlu_samples = [mmlu[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(mmlu_samples)} samples from MMLU")
    except Exception as e:
        print(f"Error extracting from MMLU: {e}")

    # 4. GPQA dataset
    try:
        gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main", cache_dir=cache_dir)
        split_name = "train" if "train" in gpqa else list(gpqa.keys())[0]
        dataset_size = len(gpqa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            gpqa_samples = [gpqa[split_name][i] for i in indices]
        else:
            gpqa_samples = [gpqa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(gpqa_samples)} samples from GPQA")
    except Exception as e:
        print(f"Error extracting from GPQA: {e}")

    # 5. MBPP dataset (loaded from HuggingFace)
    try:
        mbpp_dataset = load_dataset("mbpp", "full", cache_dir=cache_dir)
        split_name = (
            "train" if "train" in mbpp_dataset else list(mbpp_dataset.keys())[0]
        )
        mbpp_samples_all = list(mbpp_dataset[split_name])
        dataset_size = len(mbpp_samples_all)
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            mbpp_samples = [mbpp_samples_all[i] for i in indices]
        else:
            mbpp_samples = mbpp_samples_all
        print(f"Successfully extracted {len(mbpp_samples)} samples from MBPP")
    except Exception as e:
        print(f"Error extracting from MBPP: {e}")

    # 6. HumanEval dataset (loaded from HuggingFace)
    try:
        humaneval_dataset = load_dataset("openai/openai_humaneval", cache_dir=cache_dir)
        split_name = (
            "test" if "test" in humaneval_dataset else list(humaneval_dataset.keys())[0]
        )
        humaneval_samples_all = list(humaneval_dataset[split_name])
        dataset_size = len(humaneval_samples_all)
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            humaneval_samples = [humaneval_samples_all[i] for i in indices]
        else:
            humaneval_samples = humaneval_samples_all
        print(f"Successfully extracted {len(humaneval_samples)} samples from HumanEval")
    except Exception as e:
        print(f"Error extracting from HumanEval: {e}")

    # 7. GSM8K dataset
    try:
        gsm8k = load_dataset("gsm8k", "main", cache_dir=cache_dir)
        split_name = "train" if "train" in gsm8k else list(gsm8k.keys())[0]
        dataset_size = len(gsm8k[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            gsm8k_samples = [gsm8k[split_name][i] for i in indices]
        else:
            gsm8k_samples = [gsm8k[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(gsm8k_samples)} samples from GSM8K")
    except Exception as e:
        print(f"Error extracting from GSM8K: {e}")

    # 8. CommonsenseQA dataset
    try:
        commonsense_qa = load_dataset("commonsense_qa", cache_dir=cache_dir)
        split_name = (
            "train" if "train" in commonsense_qa else list(commonsense_qa.keys())[0]
        )
        dataset_size = len(commonsense_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            commonsense_qa_samples = [commonsense_qa[split_name][i] for i in indices]
        else:
            commonsense_qa_samples = [
                commonsense_qa[split_name][i] for i in range(dataset_size)
            ]
        print(
            f"Successfully extracted {len(commonsense_qa_samples)} samples from CommonsenseQA"
        )
    except Exception as e:
        print(f"Error extracting from CommonsenseQA: {e}")

    # 9. ARC-Challenge dataset
    try:
        arc_challenge = load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", cache_dir=cache_dir
        )
        split_name = (
            "train" if "train" in arc_challenge else list(arc_challenge.keys())[0]
        )
        dataset_size = len(arc_challenge[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            arc_challenge_samples = [arc_challenge[split_name][i] for i in indices]
        else:
            arc_challenge_samples = [
                arc_challenge[split_name][i] for i in range(dataset_size)
            ]
        print(
            f"Successfully extracted {len(arc_challenge_samples)} samples from ARC-Challenge"
        )
    except Exception as e:
        print(f"Error extracting from ARC-Challenge: {e}")

    # 10. OpenbookQA dataset
    try:
        openbook_qa = load_dataset("allenai/openbookqa", "main", cache_dir=cache_dir)
        split_name = "train" if "train" in openbook_qa else list(openbook_qa.keys())[0]
        dataset_size = len(openbook_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            openbook_qa_samples = [openbook_qa[split_name][i] for i in indices]
        else:
            openbook_qa_samples = [
                openbook_qa[split_name][i] for i in range(dataset_size)
            ]
        print(
            f"Successfully extracted {len(openbook_qa_samples)} samples from OpenbookQA"
        )
    except Exception as e:
        print(f"Error extracting from OpenbookQA: {e}")

    # 11. MATH dataset
    try:
        CATEGORY = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        for cate in CATEGORY:
            math = load_dataset("EleutherAI/hendrycks_math", cate, cache_dir=cache_dir)
            split_name = "train" if "train" in math else list(math.keys())[0]
            dataset_size = len(math[split_name])
            target_samples = N // len(CATEGORY) + 1
            if dataset_size >= target_samples:
                indices = random.sample(range(dataset_size), target_samples)
                math_samples.extend([math[split_name][i] for i in indices])
            else:
                math_samples.extend([math[split_name][i] for i in range(dataset_size)])
        print(f"Successfully extracted {len(math_samples)} samples from MATH")
    except Exception as e:
        print(f"Error extracting from MATH: {e}")

    return {
        "natural_qa": natural_qa_samples,
        "trivia_qa": trivia_qa_samples,
        "mmlu": mmlu_samples,
        "gpqa": gpqa_samples,
        "mbpp": mbpp_samples,
        "human_eval": humaneval_samples,
        "gsm8k": gsm8k_samples,
        "commonsense_qa": commonsense_qa_samples,
        "math": math_samples,
        "openbook_qa": openbook_qa_samples,
        "arc_challenge": arc_challenge_samples,
    }


def generate_query_data(sample_size=None, train_ratio=0.8, random_seed=42):
    """
    Generate query data from benchmark datasets.

    Args:
        sample_size: Number of samples per task
        train_ratio: Ratio for train/test split (default 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_data, test_data) - Lists of dictionaries matching query_data format
    """
    print("=== QUERY DATA GENERATION ===")

    # Use sample_size if provided, otherwise use CASE_NUM
    n_samples = sample_size if sample_size else CASE_NUM

    print(f"Extracting {n_samples} samples per task...")
    samples = get_n_samples(N=n_samples)

    data_all = []

    # Process each task type
    for task_name, task_samples in samples.items():
        if task_name == "natural_qa":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": sample["golden_answers"][0],
                    "metric": "f1_score",
                    "choices": None,
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "trivia_qa":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": sample["answer"]["normalized_aliases"][0],
                    "metric": "f1_score",
                    "choices": None,
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "mmlu":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": chr(
                        65 + sample["answer"]
                    ),  # Convert index to A, B, C, D
                    "metric": "em_mc",
                    "choices": sample["choices"],
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "gpqa":
            for sample in task_samples:
                options = [
                    sample["Correct Answer"],
                    sample["Incorrect Answer 1"],
                    sample["Incorrect Answer 2"],
                    sample["Incorrect Answer 3"],
                ]
                correct_index = 0
                mapping = list(range(len(options)))
                random.shuffle(mapping)
                new_correct_index = mapping.index(correct_index)
                shuffled_options = [
                    options[mapping.index(i)] for i in range(len(options))
                ]

                case = {
                    "task_name": task_name,
                    "query": sample["Question"],
                    "ground_truth": chr(65 + new_correct_index),
                    "metric": "em_mc",
                    "choices": shuffled_options,
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "mbpp":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["text"],
                    "ground_truth": sample["test_list"],
                    "metric": "code_eval",
                    "choices": sample["test_list"],
                    "task_id": sample["task_id"],
                }
                data_all.append(case)

        elif task_name == "human_eval":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["prompt"],
                    "ground_truth": sample["test"],
                    "metric": "code_eval",
                    "choices": None,
                    "task_id": sample["task_id"],
                }
                data_all.append(case)

        elif task_name == "gsm8k":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": sample["answer"],
                    "metric": "GSM8K",
                    "choices": None,
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "commonsense_qa":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": sample["answerKey"],
                    "metric": "em_mc",
                    "choices": sample["choices"],
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "math":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["problem"],
                    "ground_truth": sample["solution"],
                    "metric": "MATH",
                    "choices": None,
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "openbook_qa":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question_stem"],
                    "ground_truth": sample["answerKey"],
                    "metric": "em_mc",
                    "choices": sample["choices"],
                    "task_id": None,
                }
                data_all.append(case)

        elif task_name == "arc_challenge":
            for sample in task_samples:
                case = {
                    "task_name": task_name,
                    "query": sample["question"],
                    "ground_truth": sample["answerKey"],
                    "metric": "em_mc",
                    "choices": sample["choices"],
                    "task_id": None,
                }
                data_all.append(case)

    print(f"Generated {len(data_all)} base samples")

    # Convert choices to string format if needed (matching sample format)
    for case in data_all:
        if case["choices"] is not None:
            if isinstance(case["choices"], dict):
                # Already in correct format
                pass
            elif isinstance(case["choices"], list):
                # Convert to dict format like sample: {'text': [...], 'labels': [...]}
                case["choices"] = {
                    "text": case["choices"],
                    "labels": [chr(65 + i) for i in range(len(case["choices"]))],
                }

    # Split into train/test
    random.seed(random_seed)
    random.shuffle(data_all)
    train_size = int(len(data_all) * train_ratio)
    train_data = data_all[:train_size]
    test_data = data_all[train_size:]

    print(f"Split into {len(train_data)} train and {len(test_data)} test samples")

    return train_data, test_data


def save_query_data_jsonl(data_list, output_path):
    """Save query data to JSONL file matching sample format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            # Format matches sample: task_name, query, ground_truth, metric, choices, task_id
            record = {
                "task_name": item["task_name"],
                "query": item["query"],
                "ground_truth": item["ground_truth"],
                "metric": item["metric"],
                "choices": json.dumps(item["choices"])
                if item["choices"] is not None
                else None,
                "task_id": item["task_id"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(data_list)} records to {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate Query Data JSONL Files")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples per task (default: 500)",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default=None,
        help="Output path for train JSONL file",
    )
    parser.add_argument(
        "--output_test", type=str, default=None, help="Output path for test JSONL file"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run with 10 samples for quick testing"
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        loader = DataLoader(project_root)
        data_path = config.get("data_path", {})

        # Get paths from config
        output_train = loader.to_abs(data_path.get("query_data_train", ""))
        output_test = loader.to_abs(data_path.get("query_data_test", ""))

        # Get sample size from config if not provided
        if args.sample is None:
            args.sample = config.get("data_generation", {}).get("sample_size", CASE_NUM)
    else:
        # Use command-line args
        if args.output_train is None or args.output_test is None:
            parser.error(
                "Either --config or both --output_train and --output_test must be provided"
            )
        output_train = args.output_train
        output_test = args.output_test

    if args.test:
        args.sample = 10
        print("Running in test mode with 10 samples per task...")

    start_time = time.time()

    try:
        # Generate query data
        train_data, test_data = generate_query_data(
            sample_size=args.sample,
            train_ratio=config.get("data_generation", {}).get("train_ratio", 0.8),
            random_seed=config.get("data_generation", {}).get("random_seed", 42),
        )

        # Save to JSONL files
        save_query_data_jsonl(train_data, output_train)
        save_query_data_jsonl(test_data, output_test)

        total_time = time.time() - start_time
        print(
            f"\n🎉 Query data generation completed successfully in {total_time:.1f} seconds!"
        )
        print("📊 Generated data statistics:")
        print(f"  - Train samples: {len(train_data)}")
        print(f"  - Test samples: {len(test_data)}")
        print(f"  - Total samples: {len(train_data) + len(test_data)}")

        # Show sample counts by task
        all_tasks = set(item["task_name"] for item in train_data + test_data)
        print(f"  - Tasks: {sorted(all_tasks)}")

        print("\n📁 Output files:")
        print(f"  - Train: {output_train}")
        print(f"  - Test: {output_test}")

    except KeyboardInterrupt:
        print("\n⚠️  Data generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during data generation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
