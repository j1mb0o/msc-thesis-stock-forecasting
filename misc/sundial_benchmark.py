import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM

def benchmark_config(model, seqs, forecast_length, num_samples, use_cache, num_runs=10, num_warmup_runs=3):
    """
    Runs the benchmark for a single configuration and returns the mean and standard deviation.
    """
    times = []
    for i in range(num_runs + num_warmup_runs):
        start_time = time.time()
        with torch.no_grad():
            model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples, use_cache=use_cache)
        end_time = time.time()
        if i >= num_warmup_runs:
            times.append((end_time - start_time) * 1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time

if __name__ == "__main__":
    MODEL_NAME = 'thuml/sundial-base-128m'
    INPUT_LENGTH = 2880
    FORECAST_LENGTH = 64
    NUM_SAMPLES = 20
    NUM_RUNS = 10
    NUM_WARMUP_RUNS = 3

    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    seqs = torch.randn(1, INPUT_LENGTH)

    print("\n--- Benchmarking ---")
    print(f"Running {NUM_RUNS} measurements for each configuration...")

    print("\nConfiguration: KV Cache Disabled")
    mean_no_kv, std_no_kv = benchmark_config(
        model, seqs, FORECAST_LENGTH, NUM_SAMPLES, use_cache=False, num_runs=NUM_RUNS, num_warmup_runs=NUM_WARMUP_RUNS
    )

    print("\nConfiguration: KV Cache Enabled")
    mean_with_kv, std_with_kv = benchmark_config(
        model, seqs, FORECAST_LENGTH, NUM_SAMPLES, use_cache=True, num_runs=NUM_RUNS, num_warmup_runs=NUM_WARMUP_RUNS
    )

    log_file = "sundial_benchmark_summary.log"
    with open(log_file, "w") as f:
        f.write("Configuration,Mean Inference Time (ms),Standard Deviation (ms)\n")
        f.write(f"KV Cache Disabled,{mean_no_kv:.4f},{std_no_kv:.4f}\n")
        f.write(f"KV Cache Enabled,{mean_with_kv:.4f},{std_with_kv:.4f}\n")

    print("\n--- Benchmark Summary ---")
    print(f"Results logged to {log_file}")
    print("\nKV Cache Disabled:")
    print(f"  Mean: {mean_no_kv:.4f} ms")
    print(f"  Standard Deviation: {std_no_kv:.4f} ms")
    print("\nKV Cache Enabled:")
    print(f"  Mean: {mean_with_kv:.4f} ms")
    print(f"  Standard Deviation: {std_with_kv:.4f} ms")
