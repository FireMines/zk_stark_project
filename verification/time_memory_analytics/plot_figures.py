import pandas as pd
import matplotlib.pyplot as plt

# --- Plot Time Metrics using time_avg_std.csv ---
# This CSV has columns with averaged times and their std deviations.
time_df = pd.read_csv("time_avg_std.csv")

# List the time metrics you want to plot (without the _avg suffix)
# For instance: compile, setup, compute_witness, generate_proof, export_verifier.
time_metrics = ["t_compile", "t_setup", "t_compute_witness", "t_generate_proof", "t_export_verifier"]

# Create a figure for time metrics (one chart with error bars)
plt.figure(figsize=(8, 5))
x = time_df["batchsize_avg"]  # Use the averaged batchsize as the x-axis

for metric in time_metrics:
    avg_col = metric + "_avg"
    std_col = metric + "_std"
    # Plot a line for each metric with error bars.
    plt.errorbar(x, time_df[avg_col], yerr=time_df[std_col],
                 marker='o', capsize=4, label=metric)

plt.xlabel("Batch Size")
plt.ylabel("Time (seconds)")
plt.title("Time Metrics vs. Batch Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Memory Usage using final_analytics.csv ---
# This CSV has average memory usage (in MB) and their std deviations.
mem_df = pd.read_csv("final_analytics.csv")

# If you want the memory usage in GB instead of MB, convert by dividing by 1024.
mem_cols = ["max_mem_compile_avg", "max_mem_setup_avg",
            "max_mem_compute_witness_avg", "max_mem_generate_proof_avg", "max_mem_export_verifier_avg"]
std_cols = ["max_mem_compile_std", "max_mem_setup_std",
            "max_mem_compute_witness_std", "max_mem_generate_proof_std", "max_mem_export_verifier_std"]

for col in mem_cols + std_cols:
    mem_df[col] = mem_df[col] / 1024  # Convert MB to GB

# Create a figure for memory metrics.
plt.figure(figsize=(8, 5))
batchsize = mem_df["batchsize"]  # using batchsize column

# Let’s plot two groups:
# Group 1: Compilation and Setup memory usage.
plt.errorbar(batchsize, mem_df["max_mem_compile_avg"], yerr=mem_df["max_mem_compile_std"],
             marker='o', capsize=4, color='blue', label='Compilation')
plt.errorbar(batchsize, mem_df["max_mem_setup_avg"], yerr=mem_df["max_mem_setup_std"],
             marker='o', capsize=4, color='red', label='Setup')

plt.xlabel("Batch Size")
plt.ylabel("Memory usage (GB)")
plt.title("Memory usage: Compilation vs. Setup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Now, plot Group 2: Compute witness vs. Generate proof.
plt.figure(figsize=(8, 5))
plt.errorbar(batchsize, mem_df["max_mem_compute_witness_avg"], yerr=mem_df["max_mem_compute_witness_std"],
             marker='o', capsize=4, color='blue', label='Compute witness')
plt.errorbar(batchsize, mem_df["max_mem_generate_proof_avg"], yerr=mem_df["max_mem_generate_proof_std"],
             marker='o', capsize=4, color='orange', label='Generate proof')

plt.xlabel("Batch Size")
plt.ylabel("Memory usage (GB)")
plt.title("Memory usage: Compute witness vs. Generate proof")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
