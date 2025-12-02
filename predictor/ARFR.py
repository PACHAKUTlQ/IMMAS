import os
import time
import re
from typing import Dict, Tuple

from datasets import load_dataset
from openai import OpenAI
from river import tree
from tqdm import tqdm


FeatureDict = Dict[str, float]
MetricPred = Tuple[float, float]


class AgentPredictor:
    """Online predictor for latency, cost, and performance using Hoeffding trees."""

    def __init__(self, num_models: int, num_tasks: int) -> None:
        """
        Parameters
        ----------
        num_models
            Number of distinct model ids (for one-hot encoding).
        num_tasks
            Number of distinct task ids (for one-hot encoding).
        """

        self.num_models = num_models
        self.num_tasks = num_tasks

        # Use Hoeffding Tree Regressors from River for online learning.
        # One regressor per metric.
        splitter = tree.splitter.EBSTSplitter()
        self.model_latency = tree.HoeffdingTreeRegressor(splitter=splitter)
        self.model_cost = tree.HoeffdingTreeRegressor(splitter=splitter)
        self.model_perf = tree.HoeffdingTreeRegressor(splitter=splitter)

    def _make_features(
        self,
        kvmatch: float,
        model_id: int,
        task_id: int,
        prompt_length: int,
    ) -> FeatureDict:
        """
        Build a feature dictionary for River models.

        Features
        --------
        bias
            Constant 1.0.
        kvmatch
            Simulated KV-cache match ratio (0.0-1.0).
        model_*
            One-hot model id.
        task_*
            One-hot task id.
        prompt_length
            Length of the input prompt (in characters).
        """

        features: FeatureDict = {}

        features["bias"] = 1.0
        features["kvmatch"] = float(kvmatch)
        features["prompt_length"] = float(prompt_length)

        for mid in range(self.num_models):
            features[f"model_{mid}"] = 1.0 if mid == model_id else 0.0

        for tid in range(self.num_tasks):
            features[f"task_{tid}"] = 1.0 if tid == task_id else 0.0

        return features

    def predict(
        self,
        kvmatch: float,
        model_id: int,
        task_id: int,
        prompt_length: int,
    ) -> Dict[str, MetricPred]:
        """
        Predict latency, cost, and performance for the given context.

        Returns
        -------
        dict
            {
                "latency": (predicted_ms, dummy_std),
                "cost": (predicted_tokens, dummy_std),
                "performance": (predicted_prob, dummy_std),
            }

        Note
        ----
        HoeffdingTreeRegressor does not expose predictive uncertainty,
        so the second element of each tuple is set to 0.0 as a placeholder
        to keep the original interface shape.
        """

        x = self._make_features(kvmatch, model_id, task_id, prompt_length)

        lat_pred = self.model_latency.predict_one(x)
        cost_pred = self.model_cost.predict_one(x)
        perf_pred = self.model_perf.predict_one(x)

        # River regressors typically return 0.0 for untrained models;
        # handle None defensively.
        lat_pred = float(lat_pred) if lat_pred is not None else 0.0
        cost_pred = float(cost_pred) if cost_pred is not None else 0.0
        perf_pred = float(perf_pred) if perf_pred is not None else 0.0

        # Post-processing to enforce metric domains
        lat_pred = max(0.0, lat_pred)
        cost_pred = max(0.0, cost_pred)
        perf_pred = max(0.0, min(1.0, perf_pred))

        dummy_std = 0.0

        return {
            "latency": (lat_pred, dummy_std),
            "cost": (cost_pred, dummy_std),
            "performance": (perf_pred, dummy_std),
        }

    def update(
        self,
        kvmatch: float,
        model_id: int,
        task_id: int,
        prompt_length: int,
        real_latency_ms: float,
        real_cost_tokens: int,
        real_perf_correct: bool,
    ) -> None:
        """
        Update the online models with an observed data point.

        Parameters
        ----------
        kvmatch
            KV-cache match ratio (simulated).
        model_id
            Model index.
        task_id
            Task index.
        prompt_length
            Length of the input prompt (in characters).
        real_latency_ms
            Measured latency in milliseconds.
        real_cost_tokens
            Measured total tokens used.
        real_perf_correct
            True if the answer was correct, False otherwise.
        """

        x = self._make_features(kvmatch, model_id, task_id, prompt_length)

        self.model_latency.learn_one(x, float(real_latency_ms))
        self.model_cost.learn_one(x, float(real_cost_tokens))
        self.model_perf.learn_one(x, 1.0 if real_perf_correct else 0.0)


TASKS = ["elementary_mathematics", "college_computer_science", "astronomy"]
MODELS = ["placeholder-model"]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")


def format_prompt(item: Dict) -> str:
    """Format one MMLU multiple-choice question into a prompt string."""
    options = item["choices"]
    return (
        f"Question: {item['question']}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:"
    )


def main() -> None:
    """Run an online evaluation loop with River-based predictors."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 1 model, 3 task types
    predictor = AgentPredictor(num_models=len(MODELS), num_tasks=len(TASKS))

    print("Initializing HoeffdingTreeRegressor...")
    print("=" * 60)

    total_steps = 0

    for task_id, task_name in enumerate(TASKS):
        dataset = load_dataset("cais/mmlu", task_name, split="test")
        subset = dataset.select(range(5))

        print(f"\nNew task type: {task_name} (Task ID: {task_id})")

        for i, item in enumerate(tqdm(subset)):
            total_steps += 1

            mock_kvmatch = min(1.0, 0.1 * i)

            current_model_id = 0
            current_task_id = task_id

            prompt_text = format_prompt(item)
            prompt_length = len(prompt_text)

            preds = predictor.predict(
                kvmatch=mock_kvmatch,
                model_id=current_model_id,
                task_id=current_task_id,
                prompt_length=prompt_length,
            )

            print(f"\n[Step {total_steps}] Predict:")
            print(
                f"   Latency: {preds['latency'][0]
                    :.1f} ms (±{preds['latency'][1]:.1f})"
            )
            print(
                f"   Cost:    {preds['cost'][0]
                    :.1f} tokens (±{preds['cost'][1]:.1f})"
            )
            print(
                f"   Perf:    {preds['performance'][0] * 100:.1f}% "
                f"(±{preds['performance'][1]:.2f})"
            )

            start_t = time.time()
            try:
                resp = client.chat.completions.create(
                    model="placeholder-model",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer ONLY the option letter.",
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0,
                    max_tokens=1,
                )
                raw_content = resp.choices[0].message.content.strip()
                real_cost = resp.usage.total_tokens
            except Exception:
                raw_content = ""
                real_cost = 0

            real_latency = (time.time() - start_t) * 1000.0

            pred_char_match = re.search(r"[ABCD]", raw_content, re.IGNORECASE)
            pred_char = pred_char_match.group(
                0).upper() if pred_char_match else "X"
            target_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            real_answer = target_map[item["answer"]]
            real_perf = pred_char == real_answer

            print(
                f"[Step {total_steps}] "
                f"Latency={real_latency:.1f} ms, "
                f"Cost={real_cost}, "
                f"Correct={real_perf}"
            )

            predictor.update(
                kvmatch=mock_kvmatch,
                model_id=current_model_id,
                task_id=current_task_id,
                prompt_length=prompt_length,
                real_latency_ms=real_latency,
                real_cost_tokens=real_cost,
                real_perf_correct=real_perf,
            )
            print("Predictor updated\n")


if __name__ == "__main__":
    main()
