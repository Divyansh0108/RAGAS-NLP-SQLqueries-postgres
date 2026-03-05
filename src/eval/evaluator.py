import json
import time
from datetime import datetime
from pathlib import Path

from src.config import get_settings
from src.db.executor import execute_sql
from src.models.llm import CODELLAMA, QWEN, generate_sql

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()


# ── Load Examples ─────────────────────────────────────────────────────────────
def load_examples() -> list[dict]:
    examples = []
    with open(settings.examples_path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


# ── Compare Results ───────────────────────────────────────────────────────────
def results_match(ground_truth_rows: list[dict], generated_rows: list[dict]) -> bool:
    """
    Compare ground truth rows with generated rows.
    Order-insensitive comparison by converting to sets of frozensets.
    """

    def normalize(rows):
        return set(frozenset((k, str(v)) for k, v in row.items()) for row in rows)

    return normalize(ground_truth_rows) == normalize(generated_rows)


# ── Evaluate Single Example ───────────────────────────────────────────────────
def evaluate_example(example: dict, model: str) -> dict:
    question = example["question"]
    ground_truth_sql = example["sql"]
    example_id = example["id"]

    # 1. Execute ground truth SQL
    gt_execution = execute_sql(ground_truth_sql)

    # 2. Generate SQL from LLM
    start = time.time()
    try:
        llm_result = generate_sql(question, model=model)
        generated_sql = llm_result["sql"]
        latency = round(time.time() - start, 2)
        generation_error = None
    except Exception as e:
        generated_sql = ""
        latency = round(time.time() - start, 2)
        generation_error = str(e)

    # 3. Execute generated SQL
    gen_execution = (
        execute_sql(generated_sql)
        if generated_sql
        else {
            "success": False,
            "rows": [],
            "row_count": 0,
            "error": "No SQL generated",
        }
    )

    # 4. Compare results
    execution_match = (
        gt_execution["success"]
        and gen_execution["success"]
        and results_match(gt_execution["rows"], gen_execution["rows"])
    )

    return {
        "id": example_id,
        "question": question,
        "difficulty": example.get("difficulty", "unknown"),
        "model": model,
        "ground_truth_sql": ground_truth_sql,
        "generated_sql": generated_sql,
        "gt_execution_success": gt_execution["success"],
        "gen_execution_success": gen_execution["success"],
        "execution_match": execution_match,
        "gt_row_count": gt_execution["row_count"],
        "gen_row_count": gen_execution["row_count"],
        "latency_seconds": latency,
        "generation_error": generation_error,
        "gen_execution_error": gen_execution.get("error"),
    }


# ── Evaluate All Examples ─────────────────────────────────────────────────────
def evaluate_model(model: str, examples: list[dict]) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"🤖 Evaluating model: {model}")
    print(f"{'='*60}")

    results = []
    for i, example in enumerate(examples, 1):
        print(
            f"  [{i:02d}/{len(examples)}] {example['difficulty']:6s} | {example['question'][:60]}..."
        )
        result = evaluate_example(example, model=model)
        match_icon = "✅" if result["execution_match"] else "❌"
        print(
            f"           {match_icon} match={result['execution_match']} | latency={result['latency_seconds']}s"
        )
        results.append(result)

    return results


# ── Compute Metrics ───────────────────────────────────────────────────────────
def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    execution_success = sum(1 for r in results if r["gen_execution_success"])
    execution_match = sum(1 for r in results if r["execution_match"])
    avg_latency = round(sum(r["latency_seconds"] for r in results) / total, 2)

    # Per difficulty
    by_difficulty = {}
    for diff in ["simple", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_difficulty[diff] = {
                "total": len(subset),
                "execution_match": sum(1 for r in subset if r["execution_match"]),
                "accuracy": round(
                    sum(1 for r in subset if r["execution_match"]) / len(subset) * 100,
                    1,
                ),
            }

    return {
        "total": total,
        "execution_success": execution_success,
        "execution_success_rate": round(execution_success / total * 100, 1),
        "execution_match": execution_match,
        "execution_accuracy": round(execution_match / total * 100, 1),
        "avg_latency_seconds": avg_latency,
        "by_difficulty": by_difficulty,
    }


# ── Save Results ──────────────────────────────────────────────────────────────
def save_results(results: list[dict], metrics: dict, model: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace(".", "_").replace(":", "_")
    output_file = settings.evaluation_dir / f"eval_{model_slug}_{timestamp}.json"

    output = {
        "model": model,
        "timestamp": timestamp,
        "metrics": metrics,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")
    return output_file


# ── Print Summary ─────────────────────────────────────────────────────────────
def print_summary(metrics: dict, model: str):
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {model}")
    print(f"{'='*60}")
    print(f"  Total examples     : {metrics['total']}")
    print(
        f"  Execution success  : {metrics['execution_success']} ({metrics['execution_success_rate']}%)"
    )
    print(
        f"  Execution accuracy : {metrics['execution_match']} ({metrics['execution_accuracy']}%)"
    )
    print(f"  Avg latency        : {metrics['avg_latency_seconds']}s")
    print(f"\n  By difficulty:")
    for diff, stats in metrics["by_difficulty"].items():
        print(
            f"    {diff:6s} → {stats['execution_match']}/{stats['total']} ({stats['accuracy']}%)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = load_examples()
    print(f"📂 Loaded {len(examples)} examples")

    all_metrics = {}

    for model in [QWEN, CODELLAMA]:
        results = evaluate_model(model, examples)
        metrics = compute_metrics(results)
        save_results(results, metrics, model)
        print_summary(metrics, model)
        all_metrics[model] = metrics

    # ── Final Comparison ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("🏆 FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Qwen2.5-Coder':>15} {'CodeLlama':>12}")
    print(f"  {'-'*57}")
    print(
        f"  {'Execution Accuracy':<30} {all_metrics[QWEN]['execution_accuracy']:>14}% {all_metrics[CODELLAMA]['execution_accuracy']:>11}%"
    )
    print(
        f"  {'Execution Success Rate':<30} {all_metrics[QWEN]['execution_success_rate']:>14}% {all_metrics[CODELLAMA]['execution_success_rate']:>11}%"
    )
    print(
        f"  {'Avg Latency (s)':<30} {all_metrics[QWEN]['avg_latency_seconds']:>15} {all_metrics[CODELLAMA]['avg_latency_seconds']:>12}"
    )
