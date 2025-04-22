#!/usr/bin/env python3

"""
measure_stark_aggregator.py

Now that we've split aggregator main into 3 steps:
  --step=setup
  --step=witness
  --step=proof
we can measure each step's memory usage/time individually.

Usage:
  1) pip install psutil
  2) Make sure your aggregator main is compiled or we'll do it
  3) Example:
     ./measure_stark_aggregator.py --system_type=stark --clients=2
"""

import argparse
import time
import psutil
import subprocess
import csv
import os

def measure_subprocess(command):
    """
    Launch command as a subprocess,
    measure peak memory usage (MB), total time (seconds).
    Return (elapsed_seconds, peak_mem_mb).
    """
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps = psutil.Process(proc.pid)

    start = time.time()
    peak_mem = 0.0
    while proc.poll() is None:
        mem_info = ps.memory_info()
        mem_mb = mem_info.rss / (1024.0**2)
        if mem_mb > peak_mem:
            peak_mem = mem_mb
        time.sleep(0.05)

    elapsed = time.time() - start
    _stdout, _stderr = proc.communicate()

    return elapsed, peak_mem


def main():
    parser = argparse.ArgumentParser(description="Measure aggregator steps individually (setup, witness, proof).")
    parser.add_argument("--system_type", default="stark", help="System label, e.g. 'snark' or 'stark'.")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients.")
    parser.add_argument("--csv", default="aggregator_final_analytics.csv",
                        help="CSV to append results.")
    parser.add_argument("--binary", default="./target/release/stark_aggregator",
                        help="Path to aggregator binary that uses --step=setup|witness|proof.")
    args = parser.parse_args()

    system_type = args.system_type
    clients = args.clients
    csv_path = args.csv
    aggregator_bin = args.binary

    print(f"=== Measuring aggregator for system={system_type}, clients={clients} ===")

    #
    # Step 1: "compile" => cargo build --release
    #
    compile_cmd = ["cargo", "build", "--release"]
    compile_time, compile_peak_mb = measure_subprocess(compile_cmd)

    #
    # Step 2: aggregator setup => aggregator_bin --step=setup --clients=2
    #
    setup_cmd = [aggregator_bin, "--step", "setup", "--clients", str(clients)]
    setup_time, setup_peak_mb = measure_subprocess(setup_cmd)

    #
    # Step 3: aggregator witness => aggregator_bin --step=witness --clients=2
    #
    witness_cmd = [aggregator_bin, "--step", "witness", "--clients", str(clients)]
    witness_time, witness_peak_mb = measure_subprocess(witness_cmd)

    #
    # Step 4: aggregator proof => aggregator_bin --step=proof --clients=2
    #
    proof_cmd = [aggregator_bin, "--step", "proof", "--clients", str(clients)]
    proof_time, proof_peak_mb = measure_subprocess(proof_cmd)

    # Single-run => no real std dev
    compile_avg = compile_peak_mb; compile_std = 0.0
    setup_avg   = setup_peak_mb;   setup_std   = 0.0
    witness_avg = witness_peak_mb; witness_std = 0.0
    proof_avg   = proof_peak_mb;   proof_std   = 0.0

    # aggregator_final_analytics.csv columns used by aggregator_compile_setup.py / aggregator_witness_proof.py:
    header = [
        "system_type",
        "client_number",
        "max_mem_compile_avg","max_mem_compile_std",
        "max_mem_setup_avg","max_mem_setup_std",
        "max_mem_compute_witness_avg","max_mem_compute_witness_std",
        "max_mem_generate_proof_avg","max_mem_generate_proof_std"
    ]

    row = [
        system_type,
        clients,
        compile_avg, compile_std,
        setup_avg,   setup_std,
        witness_avg, witness_std,
        proof_avg,   proof_std
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Appended aggregator row: system_type={system_type}, clients={clients} in {csv_path}")
    print("Done.")

if __name__ == "__main__":
    main()
