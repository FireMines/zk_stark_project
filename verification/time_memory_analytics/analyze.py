import datetime
import os
import re
import subprocess
import time
import shutil
import tempfile

import numpy as np
import pandas as pd
import psutil
from hash import mimc_hash

# --------------------------------------------------------------------------------
# SNARK helpers (unchanged)
# --------------------------------------------------------------------------------
FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    diff = y_pred - y_true
    return np.array([int((2 * int(d)) // y_true.size) for d in diff], dtype=object)

def convert_matrix(m):
    max_field = FIELD_PRIME - 1
    arr = np.array(m, dtype=object)
    return (
        np.where(arr < 0, max_field + arr + 1, arr),
        np.where(arr > 0, 0, 1),
    )

def process_memory_usage(proc) -> tuple[list[float], float]:
    p = psutil.Process(proc.pid)
    samples = []
    while proc.poll() is None:
        try:
            samples.append(p.memory_info().rss / (1024 * 1024))
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.05)
    return samples, max(samples) if samples else 0.0

def run_process(cmd: list[str], mem_profile=True, cwd=None):
    if mem_profile:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        all_mem, max_mem = process_memory_usage(p)
        return p, all_mem, max_mem
    else:
        p = subprocess.run(cmd, capture_output=True, cwd=cwd)
        return p, None, None

def args_parser(args):
    out = []
    for arg in args:
        if isinstance(arg, (list, np.ndarray)):
            for v in np.ravel(arg):
                out.append(str(int(v) % FIELD_PRIME))
        else:
            out.append(str(int(arg) % FIELD_PRIME))
    return out

def compile_snark():
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","compile","-i","./../zokrates/root.zok"])
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return time.time()-t0, all_mem, max_mem

def setup_snark():
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","setup"])
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return time.time()-t0, all_mem, max_mem

def witness_snark(batch):
    # exactly your compute_witness body, minus printing DataFrame
    np.random.seed(0)
    precision = 1000
    ac, fe = 6, 9
    bias = (np.random.randn(ac) * precision).astype(int)
    weights = (np.random.randn(ac, fe) * precision).astype(int)
    w, w_sign = convert_matrix(weights)
    b, b_sign = convert_matrix(bias)
    X = (np.random.randn(batch, fe) * precision).astype(int)
    X, X_sign = convert_matrix(X)
    lr = 10; Y = []
    w_curr = weights.astype(object).copy()
    b_curr = bias.astype(object).copy()
    for Xi in X:
        lbl = np.random.randint(1, ac)
        y_true = np.zeros(ac, int);  y_true[lbl-1] = precision;  Y.append(lbl)
        out_layer = (w_curr.dot(Xi)//precision) + b_curr
        error = mse_prime(y_true, out_layer)
        w_curr = w_curr - ((np.outer(error,Xi)//precision)//lr)
        b_curr = b_curr - ([e//lr for e in error])
    new_w,_ = convert_matrix(np.array(w_curr,object))
    new_b,_ = convert_matrix(np.array(b_curr,object))
    ldigest = mimc_hash(new_w,new_b);  gdigest = mimc_hash(w,b)
    flat = args_parser([w,w_sign,b,b_sign,X,X_sign,Y,lr,precision,new_w,new_b,ldigest,gdigest])
    # debug: print(f"SNARK args {len(flat)}")
    cmd = ["zokrates","compute-witness","--verbose","-a"] + flat
    t0 = time.time()
    p, all_mem, max_mem = run_process(cmd)
    dt = time.time()-t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode() or p.stdout.read().decode())
    return dt, all_mem, max_mem

def prove_snark():
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","generate-proof"])
    dt = time.time()-t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return dt, all_mem, max_mem

def export_snark():
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","export-verifier"])
    dt = time.time()-t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return dt, all_mem, max_mem

# --------------------------------------------------------------------------------
# STARK helpers (Rust)
# --------------------------------------------------------------------------------
def create_stark_test_data(num_devices: int, samples_per_device: int = 50):
    """Create test data for STARK implementation similar to the setup"""
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="stark_test_data_")
    
    np.random.seed(0)  # For reproducibility
    ac, fe = 6, 9
    
    for device_id in range(num_devices):
        device_dir = os.path.join(temp_dir, f"Device_{device_id + 1}")
        os.makedirs(device_dir, exist_ok=True)
        
        # Generate random data similar to the format expected by EdgeDevice
        # Create features (fe columns) and labels (1 column)
        features = np.random.randn(samples_per_device, fe)
        labels = np.random.randint(1, ac + 1, samples_per_device)
        
        # Combine features and labels into a single CSV
        data = np.column_stack([features, labels])
        
        # Save as device_data.txt
        device_file = os.path.join(device_dir, "device_data.txt")
        np.savetxt(device_file, data, delimiter=',', fmt='%.6f')
    
    return temp_dir

def build_stark():
    """Build the Rust STARK implementation"""
    t0 = time.time()
    # Build the Rust project
    p, all_mem, max_mem = run_process(["cargo", "build", "--release"], cwd="..")
    if p.returncode != 0:
        raise RuntimeError(f"STARK build failed:\n{p.stderr.read().decode()}")
    return time.time() - t0, all_mem, max_mem

def run_stark_step(step: str, batch_size: int, data_dir: str, rust_binary: str):
    """Run a specific STARK step"""
    t0 = time.time()
    cmd = [
        rust_binary,
        "--step", step,
        "--bs", str(batch_size),
        "--data-dir", data_dir
    ]
    p, all_mem, max_mem = run_process(cmd)
    dt = time.time() - t0
    if p.returncode != 0:
        error_msg = p.stderr.read().decode() if p.stderr else ""
        stdout_msg = p.stdout.read().decode() if p.stdout else ""
        raise RuntimeError(f"STARK {step} failed:\n{error_msg}\n{stdout_msg}")
    return dt, all_mem, max_mem

def setup_stark_environment(batch_size: int, num_devices: int = 3):
    """Set up the environment for STARK testing"""
    # Create test data
    data_dir = create_stark_test_data(num_devices)
    
    # Build the STARK binary
    build_time, build_mem_samples, build_max_mem = build_stark()
    
    # Find the binary path
    rust_binary = "../target/release/zk_stark_project"
    if not os.path.exists(rust_binary):
        # Try alternative names
        alternative_paths = [
            "../target/release/stark_aggregator",
            "../target/release/main",
            "../target/release/zk-stark-project"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                rust_binary = alt_path
                break
        else:
            raise RuntimeError(f"STARK binary not found. Expected at {rust_binary}")
    
    return data_dir, rust_binary, build_time, build_max_mem

def run_stark_benchmark(batch_size: int):
    """Run complete STARK benchmark"""
    print(f"\nRunning STARK benchmark with batch size {batch_size}...")
    
    # Setup environment
    data_dir, rust_binary, build_time, build_max_mem = setup_stark_environment(batch_size)
    
    try:
        # Run the different steps
        # Note: Your main.rs seems to do everything in one run, so we'll measure the total time
        total_time, total_mem_samples, total_max_mem = run_stark_step("setup", batch_size, data_dir, rust_binary)
        witness_time, witness_mem_samples, witness_max_mem = run_stark_step("witness", batch_size, data_dir, rust_binary)
        proof_time, proof_mem_samples, proof_max_mem = run_stark_step("proof", batch_size, data_dir, rust_binary)
        
        return {
            "build_time": build_time,
            "build_max_mem": build_max_mem,
            "setup_time": total_time,
            "setup_max_mem": total_max_mem,
            "witness_time": witness_time,
            "witness_max_mem": witness_max_mem,
            "proof_time": proof_time,
            "proof_max_mem": proof_max_mem
        }
    
    finally:
        # Cleanup test data
        shutil.rmtree(data_dir, ignore_errors=True)

# --------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read batch size from root.zok
    batch = 1
    try:
        with open("../zokrates/root.zok") as f:
            for L in f:
                m = re.search(r"const\s+u32\s+bs\s*=\s*(\d+);", L)
                if m:
                    batch = int(m.group(1))
    except FileNotFoundError:
        print("Warning: Could not find ../zokrates/root.zok, using default batch size = 1")
    
    print(f"Batch size = {batch}")

    # DataFrame columns
    cols = [
        "batch",
        "snark_t_compile", "snark_t_setup", "snark_t_witness", "snark_t_proof", "snark_t_export",
        "snark_mem_compile", "snark_mem_setup", "snark_mem_witness", "snark_mem_proof", "snark_mem_export",
        "stark_t_build", "stark_t_setup", "stark_t_witness", "stark_t_proof",
        "stark_mem_build", "stark_mem_setup", "stark_mem_witness", "stark_mem_proof"
    ]
    df = pd.DataFrame(columns=cols)

    # Initialize row with defaults
    row_data = {"batch": batch}
    
    # Run zk-SNARK
    print("\nRunning zk-SNARK benchmark...")
    try:
        t1, m1a, m1b = compile_snark()
        t2, m2a, m2b = setup_snark()
        t3, m3a, m3b = witness_snark(batch)
        t4, m4a, m4b = prove_snark()
        t5, m5a, m5b = export_snark()
        
        # Add SNARK metrics to row
        row_data.update({
            "snark_t_compile": t1, "snark_t_setup": t2, "snark_t_witness": t3, 
            "snark_t_proof": t4, "snark_t_export": t5,
            "snark_mem_compile": m1b, "snark_mem_setup": m2b, "snark_mem_witness": m3b, 
            "snark_mem_proof": m4b, "snark_mem_export": m5b
        })
        print("✓ zk-SNARK benchmark completed successfully")
    except Exception as e:
        print(f"✗ zk-SNARK benchmark failed: {e}")
        # Fill with NaN for failed SNARK metrics
        for col in cols:
            if col.startswith("snark_"):
                row_data[col] = np.nan

    # Run zk-STARK
    try:
        stark_metrics = run_stark_benchmark(batch)
        
        # Add STARK metrics to row
        row_data.update({
            "stark_t_build": stark_metrics["build_time"],
            "stark_t_setup": stark_metrics["setup_time"],
            "stark_t_witness": stark_metrics["witness_time"],
            "stark_t_proof": stark_metrics["proof_time"],
            "stark_mem_build": stark_metrics["build_max_mem"],
            "stark_mem_setup": stark_metrics["setup_max_mem"],
            "stark_mem_witness": stark_metrics["witness_max_mem"],
            "stark_mem_proof": stark_metrics["proof_max_mem"]
        })
        print("✓ zk-STARK benchmark completed successfully")
    except Exception as e:
        print(f"✗ zk-STARK benchmark failed: {e}")
        # Fill with NaN for failed STARK metrics
        for col in cols:
            if col.startswith("stark_"):
                row_data[col] = np.nan

    # Append row to DataFrame
    df = df.append(row_data, ignore_index=True)

    # Write results
    out = "unified_metrics.csv"
    if os.path.exists(out):
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)

    print(f"\nResults written to {out}")
    print("\nSummary:")
    print("=" * 50)
    
    # Print summary of metrics
    if not np.isnan(row_data.get("snark_t_compile", np.nan)):
        print("zk-SNARK metrics:")
        print(f"  Compile: {row_data['snark_t_compile']:.3f}s ({row_data['snark_mem_compile']:.1f}MB)")
        print(f"  Setup:   {row_data['snark_t_setup']:.3f}s ({row_data['snark_mem_setup']:.1f}MB)")
        print(f"  Witness: {row_data['snark_t_witness']:.3f}s ({row_data['snark_mem_witness']:.1f}MB)")
        print(f"  Proof:   {row_data['snark_t_proof']:.3f}s ({row_data['snark_mem_proof']:.1f}MB)")
        print(f"  Export:  {row_data['snark_t_export']:.3f}s ({row_data['snark_mem_export']:.1f}MB)")
    
    if not np.isnan(row_data.get("stark_t_build", np.nan)):
        print("\nzk-STARK metrics:")
        print(f"  Build:   {row_data['stark_t_build']:.3f}s ({row_data['stark_mem_build']:.1f}MB)")
        print(f"  Setup:   {row_data['stark_t_setup']:.3f}s ({row_data['stark_mem_setup']:.1f}MB)")
        print(f"  Witness: {row_data['stark_t_witness']:.3f}s ({row_data['stark_mem_witness']:.1f}MB)")
        print(f"  Proof:   {row_data['stark_t_proof']:.3f}s ({row_data['stark_mem_proof']:.1f}MB)")
    
    print("=" * 50)
    print(f"Unified metrics saved to {out} with SNARK and STARK comparisons")