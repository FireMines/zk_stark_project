import datetime
import os
import re
import subprocess
import time
import shutil
import tempfile
import json
import threading
import queue
import psutil
import argparse
import statistics

import numpy as np
import pandas as pd
from hash import mimc_hash

# Add argument parser for configuration
parser = argparse.ArgumentParser(description='Run zk-SNARK and zk-STARK benchmarks')
parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs to perform (default: 3)')
parser.add_argument('--batch-size', type=int, default=None, help='Override batch size (otherwise read from root.zok)')
parser.add_argument('--skip-snark', action='store_true', help='Skip SNARK benchmarks')
parser.add_argument('--skip-stark', action='store_true', help='Skip STARK benchmarks')
parser.add_argument('--output-prefix', type=str, default='unified_metrics', help='Output file prefix (default: unified_metrics)')

# --------------------------------------------------------------------------------
# Enhanced Memory Monitoring System
# --------------------------------------------------------------------------------

class RobustMemoryMonitor:
    def __init__(self, process, poll_interval=0.2):
        self.process = process
        self.poll_interval = poll_interval
        self.samples = []
        self.max_memory = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)  # Wait max 1 second
    
    def _monitor_memory(self):
        """Internal memory monitoring method"""
        try:
            while self.monitoring and self.process.poll() is None:
                try:
                    # Create a new psutil process object to avoid caching issues
                    p = psutil.Process(self.process.pid)
                    memory_mb = p.memory_info().rss / (1024 * 1024)
                    self.samples.append(memory_mb)
                    self.max_memory = max(self.max_memory, memory_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                except Exception as e:
                    # print(f"DEBUG: Memory monitoring error: {e}")
                    break
                
                time.sleep(self.poll_interval)
        except:
            pass
    
    def get_results(self):
        """Get monitoring results"""
        self.stop_monitoring()
        return self.samples, self.max_memory

def monitor_process_memory_robust(cmd, cwd=None):
    """Monitor process memory with robust error handling"""
    # print(f"DEBUG: Starting robust memory monitoring for: {' '.join(cmd)}")
    
    # Start the process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    
    # Start memory monitoring
    monitor = RobustMemoryMonitor(process)
    monitor.start_monitoring()
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    
    # Get memory results
    samples, max_memory = monitor.get_results()
    
    # print(f"DEBUG: Process completed. Max memory: {max_memory:.1f} MB, Samples: {len(samples)}")
    
    # Create result object similar to subprocess.run
    class ProcessResult:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
            self.stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
    
    return ProcessResult(process.returncode, stdout, stderr), samples, max_memory

def run_with_external_memory_monitoring(cmd, cwd=None):
    """Use external tools to monitor memory if available"""
    try:
        # Try using /usr/bin/time if available (Unix/Linux systems)
        if shutil.which("time"):
            # Use GNU time for memory monitoring
            time_cmd = ["time", "-v"] + cmd
            process = subprocess.run(time_cmd, capture_output=True, text=True, cwd=cwd)
            
            # Parse time output for memory info
            max_memory = 0.0
            if process.stderr:
                for line in process.stderr.split('\n'):
                    if "Maximum resident set size" in line:
                        # Extract memory in KB and convert to MB
                        match = re.search(r'(\d+)', line)
                        if match:
                            max_memory = int(match.group(1)) / 1024.0
                            break
            
            return process, None, max_memory
        else:
            # Fallback to regular execution without memory monitoring
            # print("DEBUG: External time command not available, running without memory monitoring")
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
            return process, None, 0.0
    except Exception as e:
        # print(f"DEBUG: External monitoring failed: {e}")
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        return process, None, 0.0

# --------------------------------------------------------------------------------
# SNARK helpers (enhanced with proof size tracking)
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
    """Original memory monitoring (kept for SNARK compatibility)"""
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
    """Original run_process (kept for SNARK compatibility)"""
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

def compile_snark(batch_size=None):
    # If batch_size is provided, update the root.zok file
    if batch_size is not None:
        update_zokrates_batch_size(batch_size)
    
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","compile","-i","./../zokrates/root.zok"])
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return time.time()-t0, all_mem, max_mem

def update_zokrates_batch_size(batch_size):
    """Update the batch size in root.zok file"""
    root_zok_path = "../zokrates/root.zok"
    
    # Read the current file
    if not os.path.exists(root_zok_path):
        raise RuntimeError(f"Cannot find {root_zok_path}")
    
    with open(root_zok_path, 'r') as f:
        lines = f.readlines()
    
    # Update the batch size line
    updated_lines = []
    found_bs = False
    
    for line in lines:
        if re.match(r"const\s+u32\s+bs\s*=", line):
            updated_lines.append(f"const u32 bs = {batch_size};\n")
            found_bs = True
        else:
            updated_lines.append(line)
    
    if not found_bs:
        raise RuntimeError(f"Could not find batch size declaration in {root_zok_path}")
    
    # Write back the updated file
    with open(root_zok_path, 'w') as f:
        f.writelines(updated_lines)
    
    # print(f"DEBUG: Updated batch size in {root_zok_path} to {batch_size}")

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
    
    # Get proof size
    proof_size = 0
    if os.path.exists("proof.json"):
        proof_size = os.path.getsize("proof.json")
    
    return dt, all_mem, max_mem, proof_size

def export_snark():
    t0 = time.time()
    p, all_mem, max_mem = run_process(["zokrates","export-verifier"])
    dt = time.time()-t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr.read().decode())
    return dt, all_mem, max_mem

# --------------------------------------------------------------------------------
# STARK helpers with enhanced memory monitoring
# --------------------------------------------------------------------------------

def find_stark_binary():
    """Find the correct STARK binary with better path resolution"""
    # print("DEBUG: Searching for STARK binary...")
    
    # Check based on Cargo.toml
    possible_names = []
    try:
        # Try to read the binary name from Cargo.toml
        with open("../Cargo.toml", "r") as f:
            content = f.read()
            if 'name = "zk_stark_project"' in content:
                possible_names.append("zk_stark_project")
            elif 'name = "main"' in content:
                possible_names.append("main")
        
        # Also check [[bin]] sections
        import re
        bin_matches = re.findall(r'\[\[bin\]\]\s*name\s*=\s*"([^"]+)"', content)
        possible_names.extend(bin_matches)
    except:
        pass
    
    # Default names to try
    possible_names.extend(["zk_stark_project", "main", "stark_aggregator", "zk-stark-project"])
    
    # Possible paths
    possible_dirs = [
        "../target/release/",
        "../../target/release/", 
        "./target/release/",
        "target/release/",
    ]
    
    # Remove duplicates while preserving order
    possible_names = list(dict.fromkeys(possible_names))
    
    # print(f"DEBUG: Looking for binaries: {possible_names}")
    # print(f"DEBUG: In directories: {possible_dirs}")
    
    for dir_path in possible_dirs:
        for name in possible_names:
            full_path = os.path.join(dir_path, name)
            if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                # print(f"DEBUG: Found STARK binary at {full_path}")
                return full_path
    
    raise RuntimeError(f"STARK binary not found. Tried names: {possible_names} in dirs: {possible_dirs}")

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
    """Build the Rust STARK implementation with robust memory monitoring"""
    # print("DEBUG: Building STARK...")
    # print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    # Check if Cargo.toml exists in parent directory
    if os.path.exists("../Cargo.toml"):
        build_dir = ".."
    elif os.path.exists("Cargo.toml"):
        build_dir = "."
    else:
        # Try to find Cargo.toml
        for root, dirs, files in os.walk("../.."):
            if "Cargo.toml" in files and "zk_stark_project" in open(os.path.join(root, "Cargo.toml")).read():
                build_dir = root
                break
        else:
            raise RuntimeError("Cannot find Cargo.toml for STARK project")
    
    # print(f"DEBUG: Building in directory: {build_dir}")
    
    # Build with enhanced memory monitoring
    try:
        # print("DEBUG: Using robust memory monitoring for build...")
        p, all_mem, max_mem = monitor_process_memory_robust(["cargo", "build", "--release"], cwd=build_dir)
    except Exception as e:
        # print(f"DEBUG: Robust monitoring failed for build: {e}")
        # Fallback to regular build
        p = subprocess.run(["cargo", "build", "--release"], capture_output=True, text=True, cwd=build_dir)
        all_mem, max_mem = None, 0.0
    
    dt = (time.time() - time.time())  # Will be calculated properly in the calling function
    
    if p.returncode != 0:
        error_msg = p.stderr if p.stderr else ""
        stdout_msg = p.stdout if p.stdout else ""
        # print(f"DEBUG: Build failed with return code {p.returncode}")
        # print(f"DEBUG: Build stderr: {error_msg}")
        # print(f"DEBUG: Build stdout: {stdout_msg}")
        raise RuntimeError(f"STARK build failed:\n{error_msg}\n{stdout_msg}")
    
    # print(f"DEBUG: Build completed")
    # if max_mem:
    #     print(f"DEBUG: Build max memory: {max_mem:.1f} MB")
    
    return 0.0, all_mem, max_mem  # Time will be calculated in setup function

def run_stark_step_with_details_enhanced(step: str, batch_size: int, data_dir: str, rust_binary: str, enable_memory_profiling=True):
    """Run a specific STARK step with enhanced memory monitoring that won't hang"""
    # print(f"DEBUG: {'Enabling' if enable_memory_profiling else 'Disabling'} enhanced memory profiling for {step} step")
    
    t0 = time.time()
    cmd = [
        rust_binary,
        "--step", step,
        "--bs", str(batch_size),
        "--data-dir", data_dir,
        "--verbose"
    ]
    
    # Use enhanced memory monitoring
    if enable_memory_profiling:
        # print(f"DEBUG: Running with enhanced memory profiling: {' '.join(cmd)}")
        try:
            # Try robust monitoring first
            # print("DEBUG: Attempting robust memory monitoring...")
            p, all_mem, max_mem = monitor_process_memory_robust(cmd)
        except Exception as e:
            # print(f"DEBUG: Robust monitoring failed: {e}")
            # Fallback to external monitoring
            try:
                # print("DEBUG: Falling back to external monitoring...")
                p, all_mem, max_mem = run_with_external_memory_monitoring(cmd)
            except Exception as e:
                # print(f"DEBUG: External monitoring failed: {e}")
                # Final fallback: run without memory monitoring
                # print("DEBUG: Running without memory monitoring as fallback")
                p = subprocess.run(cmd, capture_output=True, text=True)
                all_mem, max_mem = None, 0.0
    else:
        # print(f"DEBUG: Running without memory profiling: {' '.join(cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True)
        all_mem, max_mem = None, 0.0
    
    dt = time.time() - t0
    
    # Debug output
    # print(f"DEBUG: Command executed in {dt:.3f}s")
    # print(f"DEBUG: Return code: {p.returncode}")
    # if enable_memory_profiling and max_mem:
    #     print(f"DEBUG: Max memory usage: {max_mem:.1f} MB")
    #     if all_mem:
    #         print(f"DEBUG: Memory samples collected: {len(all_mem)}")
    
    # if p.stdout:
    #     print(f"DEBUG: Stdout sample: {p.stdout[:500]}...")
    # if p.stderr and p.stderr.strip():
    #     print(f"DEBUG: Stderr sample: {p.stderr[:500]}...")
    
    # Parse output for proof size if available
    proof_size = None
    total_proof_size = None
    
    if p.returncode == 0:
        output = p.stdout if p.stdout else ""
        
        # Look for different proof size patterns
        patterns = [
            r'proof:\s*\d+ms,\s*(\d+)\s*bytes',
            r'Proof size:\s*(\d+)\s*bytes',
            r'Total proof size:\s*(\d+)\s*bytes',
            r'Aggregation proof size:\s*(\d+)\s*bytes',
            r'Training proof size:\s*(\d+)\s*bytes',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                if "total" in pattern.lower():
                    total_proof_size = int(match.group(1))
                elif "training" in pattern.lower() and step == "proof":
                    proof_size = int(match.group(1))
                else:
                    proof_size = int(match.group(1))
                # print(f"DEBUG: Found proof size {match.group(1)} with pattern: {pattern}")
    
    if p.returncode != 0:
        error_msg = p.stderr if p.stderr else ""
        stdout_msg = p.stdout if p.stdout else ""
        raise RuntimeError(f"STARK {step} failed (return code {p.returncode}):\nSTDERR:\n{error_msg}\nSTDOUT:\n{stdout_msg}")
    
    # Return total proof size if available, otherwise individual proof size
    final_proof_size = total_proof_size if total_proof_size is not None else proof_size
    
    # Ensure max_mem is a float (could be None)
    max_mem = float(max_mem) if max_mem is not None else 0.0
    
    return dt, all_mem, max_mem, final_proof_size

def setup_stark_environment(batch_size: int, num_devices: int = 3):
    """Set up the environment for STARK testing"""
    # Print current working directory for debugging
    # print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    # Don't create new test data - use existing data structure
    # Check if the default data directory exists
    possible_data_dirs = [
        "../devices/edge_device/data",
        "devices/edge_device/data",
        "../../devices/edge_device/data",
        "../zk_stark_project/devices/edge_device/data",
    ]
    
    data_dir = None
    for d in possible_data_dirs:
        if os.path.exists(d):
            data_dir = d
            # print(f"DEBUG: Using existing data directory: {data_dir}")
            # Check if it has Device_* subdirectories
            try:
                device_dirs = [f for f in os.listdir(d) if f.startswith("Device_") and os.path.isdir(os.path.join(d, f))]
                # print(f"DEBUG: Found {len(device_dirs)} device directories: {device_dirs}")
            except:
                # print(f"DEBUG: Error reading directory {d}")
                continue
            break
    
    if data_dir is None:
        # Only create test data if no existing data is found
        print("No existing data directory found, creating test data")
        data_dir = create_stark_test_data(num_devices)
    
    # Build the STARK binary with enhanced memory monitoring
    print("Building STARK binary...")
    t0 = time.time()
    build_time, build_mem_samples, build_max_mem = build_stark()
    build_time = time.time() - t0  # Actually calculate the build time
    
    # Find the binary using improved method
    rust_binary = find_stark_binary()
    
    return data_dir, rust_binary, build_time, build_max_mem

def run_stark_benchmark_enhanced(batch_size: int):
    """Run complete STARK benchmark with enhanced memory monitoring"""
    print(f"Running STARK benchmark with batch size {batch_size}...")
    
    # Setup environment
    data_dir, rust_binary, build_time, build_max_mem = setup_stark_environment(batch_size)
    
    # Make paths absolute to avoid issues
    rust_binary = os.path.abspath(rust_binary)
    data_dir = os.path.abspath(data_dir)
    
    # print(f"DEBUG: Using binary: {rust_binary}")
    # print(f"DEBUG: Using data directory: {data_dir}")
    # print(f"DEBUG: Binary exists: {os.path.exists(rust_binary)}")
    # print(f"DEBUG: Data directory exists: {os.path.exists(data_dir)}")
    
    try:
        # Run all steps with enhanced memory monitoring enabled
        print("Running setup step...")
        setup_time, setup_mem_samples, setup_max_mem, _ = run_stark_step_with_details_enhanced(
            "setup", batch_size, data_dir, rust_binary, enable_memory_profiling=True)
        
        print("Running witness step...")
        witness_time, witness_mem_samples, witness_max_mem, _ = run_stark_step_with_details_enhanced(
            "witness", batch_size, data_dir, rust_binary, enable_memory_profiling=True)
        
        print("Running proof step...")
        proof_time, proof_mem_samples, proof_max_mem, stark_proof_size = run_stark_step_with_details_enhanced(
            "proof", batch_size, data_dir, rust_binary, enable_memory_profiling=True)
        
        results = {
            "build_time": build_time,
            "build_max_mem": build_max_mem if build_max_mem else 0.0,
            "setup_time": setup_time,
            "setup_max_mem": setup_max_mem if setup_max_mem else 0.0,
            "witness_time": witness_time,
            "witness_max_mem": witness_max_mem if witness_max_mem else 0.0,
            "proof_time": proof_time,
            "proof_max_mem": proof_max_mem if proof_max_mem else 0.0,
            "proof_size": stark_proof_size if stark_proof_size is not None else 0
        }
        
        # print(f"DEBUG: STARK results with memory measurements: {results}")
        return results
    
    finally:
        # Only cleanup if we created temporary data
        if "stark_test_data" in data_dir:
            # print(f"DEBUG: Cleaning up temporary data directory: {data_dir}")
            shutil.rmtree(data_dir, ignore_errors=True)
        # else:
        #     print(f"DEBUG: Keeping existing data directory: {data_dir}")

# --------------------------------------------------------------------------------
# Statistics and Analysis Functions
# --------------------------------------------------------------------------------

def calculate_statistics(values):
    """Calculate statistics for a list of values, handling NaN values"""
    clean_values = [v for v in values if not np.isnan(v)]
    if not clean_values:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'cv': np.nan,  # Coefficient of variation
            'median': np.nan
        }
    
    return {
        'mean': statistics.mean(clean_values),
        'std': statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0,
        'min': min(clean_values),
        'max': max(clean_values),
        'cv': statistics.stdev(clean_values) / statistics.mean(clean_values) * 100 if len(clean_values) > 1 and statistics.mean(clean_values) != 0 else 0.0,
        'median': statistics.median(clean_values)
    }

def detect_outliers(values, threshold=2.0):
    """Detect outliers using z-score method"""
    clean_values = [v for v in values if not np.isnan(v)]
    if len(clean_values) < 3:
        return []
    
    mean = statistics.mean(clean_values)
    std = statistics.stdev(clean_values)
    
    if std == 0:
        return []
    
    outliers = []
    for i, v in enumerate(values):
        if not np.isnan(v):
            z_score = abs(v - mean) / std
            if z_score > threshold:
                outliers.append(i + 1)
    
    return outliers

def print_multi_run_analysis(df, metric_name, runs):
    """Print detailed analysis for multiple runs of a metric"""
    values = [df.iloc[i][metric_name] for i in range(runs)]
    stats = calculate_statistics(values)
    outliers = detect_outliers(values)
    
    print(f"\n{metric_name}:")
    print(f"  Values: {[f'{v:.3f}' if not np.isnan(v) else 'Failed' for v in values]}")
    if not np.isnan(stats['mean']):
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  CV: {stats['cv']:.1f}%")
        if outliers:
            print(f"  Outliers (runs): {outliers}")
    else:
        print("  All runs failed")

def run_snark_benchmark(batch):
    """Run full SNARK benchmark and return results"""
    results = {}
    
    try:
        t1, m1a, m1b = compile_snark(batch)  # Pass batch size to compilation
        results.update({
            "snark_t_compile": t1,
            "snark_mem_compile": m1b
        })
    except Exception as e:
        print(f"SNARK compile failed: {e}")
        results.update({
            "snark_t_compile": np.nan,
            "snark_mem_compile": np.nan
        })
    
    try:
        t2, m2a, m2b = setup_snark()
        results.update({
            "snark_t_setup": t2,
            "snark_mem_setup": m2b
        })
    except Exception as e:
        print(f"SNARK setup failed: {e}")
        results.update({
            "snark_t_setup": np.nan,
            "snark_mem_setup": np.nan
        })
    
    try:
        t3, m3a, m3b = witness_snark(batch)
        results.update({
            "snark_t_witness": t3,
            "snark_mem_witness": m3b
        })
    except Exception as e:
        print(f"SNARK witness failed: {e}")
        results.update({
            "snark_t_witness": np.nan,
            "snark_mem_witness": np.nan
        })
    
    try:
        t4, m4a, m4b, snark_proof_size = prove_snark()
        results.update({
            "snark_t_proof": t4,
            "snark_mem_proof": m4b,
            "snark_proof_size": snark_proof_size
        })
    except Exception as e:
        print(f"SNARK prove failed: {e}")
        results.update({
            "snark_t_proof": np.nan,
            "snark_mem_proof": np.nan,
            "snark_proof_size": np.nan
        })
    
    try:
        t5, m5a, m5b = export_snark()
        results.update({
            "snark_t_export": t5,
            "snark_mem_export": m5b
        })
    except Exception as e:
        print(f"SNARK export failed: {e}")
        results.update({
            "snark_t_export": np.nan,
            "snark_mem_export": np.nan
        })
    
    return results

def run_stark_benchmark(batch):
    """Run full STARK benchmark and return results"""
    try:
        stark_metrics = run_stark_benchmark_enhanced(batch)
        return {
            "stark_t_build": stark_metrics["build_time"],
            "stark_t_setup": stark_metrics["setup_time"],
            "stark_t_witness": stark_metrics["witness_time"],
            "stark_t_proof": stark_metrics["proof_time"],
            "stark_mem_build": stark_metrics["build_max_mem"],
            "stark_mem_setup": stark_metrics["setup_max_mem"],
            "stark_mem_witness": stark_metrics["witness_max_mem"],
            "stark_mem_proof": stark_metrics["proof_max_mem"],
            "stark_proof_size": stark_metrics["proof_size"]
        }
    except Exception as e:
        print(f"STARK benchmark failed: {e}")
        return {
            "stark_t_build": np.nan,
            "stark_t_setup": np.nan,
            "stark_t_witness": np.nan,
            "stark_t_proof": np.nan,
            "stark_mem_build": np.nan,
            "stark_mem_setup": np.nan,
            "stark_mem_witness": np.nan,
            "stark_mem_proof": np.nan,
            "stark_proof_size": np.nan
        }

# --------------------------------------------------------------------------------
# Main driver with enhanced analysis
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Read batch size from root.zok or use command line argument
    batch = args.batch_size
    if batch is None:
        batch = 1
        try:
            with open("../zokrates/root.zok") as f:
                for L in f:
                    m = re.search(r"const\s+u32\s+bs\s*=\s*(\d+);", L)
                    if m:
                        batch = int(m.group(1))
        except FileNotFoundError:
            print("Warning: Could not find ../zokrates/root.zok, using default batch size = 1")
    
    print(f"Running {args.runs} benchmark runs with batch size = {batch}")

    # Enhanced DataFrame columns including proof sizes
    cols = [
        "batch", "run",
        "snark_t_compile", "snark_t_setup", "snark_t_witness", "snark_t_proof", "snark_t_export",
        "snark_mem_compile", "snark_mem_setup", "snark_mem_witness", "snark_mem_proof", "snark_mem_export",
        "snark_proof_size",
        "stark_t_build", "stark_t_setup", "stark_t_witness", "stark_t_proof",
        "stark_mem_build", "stark_mem_setup", "stark_mem_witness", "stark_mem_proof",
        "stark_proof_size"
    ]
    df = pd.DataFrame(columns=cols)

    # Run multiple benchmark iterations
    print("\n" + "="*80)
    print(f"Starting {args.runs} benchmark runs...")
    print("="*80)
    
    for run_num in range(1, args.runs + 1):
        print(f"\n{'='*20} Run {run_num}/{args.runs} {'='*20}")
        
        # Initialize row with defaults
        row_data = {"batch": batch, "run": run_num}
        
        # Run SNARK benchmark if not skipped
        if not args.skip_snark:
            print("\nRunning zk-SNARK benchmark...")
            snark_results = run_snark_benchmark(batch)
            row_data.update(snark_results)
            
            if not np.isnan(snark_results.get("snark_t_proof", np.nan)):
                print("✓ zk-SNARK benchmark completed successfully")
            else:
                print("✗ zk-SNARK benchmark failed")
        else:
            print("Skipping SNARK benchmark")
            for col in cols:
                if col.startswith("snark_"):
                    row_data[col] = np.nan
        
        # Run STARK benchmark if not skipped
        if not args.skip_stark:
            print("\nRunning zk-STARK benchmark...")
            stark_results = run_stark_benchmark(batch)
            row_data.update(stark_results)
            
            if not np.isnan(stark_results.get("stark_t_proof", np.nan)):
                print("✓ zk-STARK benchmark completed successfully")
            else:
                print("✗ zk-STARK benchmark failed")
        else:
            print("Skipping STARK benchmark")
            for col in cols:
                if col.startswith("stark_"):
                    row_data[col] = np.nan
        
        # Append row to DataFrame
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Show progress
        print(f"\nRun {run_num} completed. Total time: {time.time() - time.time():.1f}s")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_prefix}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults written to {output_file}")

    # Multi-run analysis
    print("\n" + "="*80)
    print("MULTI-RUN ANALYSIS")
    print("="*80)
    
    # Time metrics analysis
    print("\nTime Metrics Analysis:")
    print("-" * 40)
    
    if not args.skip_snark:
        print("\nSNARK Time Metrics:")
        for metric in ["snark_t_compile", "snark_t_setup", "snark_t_witness", "snark_t_proof", "snark_t_export"]:
            print_multi_run_analysis(df, metric, args.runs)
    
    if not args.skip_stark:
        print("\nSTARK Time Metrics:")
        for metric in ["stark_t_build", "stark_t_setup", "stark_t_witness", "stark_t_proof"]:
            print_multi_run_analysis(df, metric, args.runs)
    
    # Memory metrics analysis
    print("\nMemory Metrics Analysis:")
    print("-" * 40)
    
    if not args.skip_snark:
        print("\nSNARK Memory Metrics (MB):")
        for metric in ["snark_mem_compile", "snark_mem_setup", "snark_mem_witness", "snark_mem_proof", "snark_mem_export"]:
            print_multi_run_analysis(df, metric, args.runs)
    
    if not args.skip_stark:
        print("\nSTARK Memory Metrics (MB):")
        for metric in ["stark_mem_build", "stark_mem_setup", "stark_mem_witness", "stark_mem_proof"]:
            print_multi_run_analysis(df, metric, args.runs)
    
    # Proof size analysis
    print("\nProof Size Analysis:")
    print("-" * 40)
    
    if not args.skip_snark:
        print_multi_run_analysis(df, "snark_proof_size", args.runs)
    if not args.skip_stark:
        print_multi_run_analysis(df, "stark_proof_size", args.runs)
    
    # Overall comparison across runs
    if not args.skip_snark and not args.skip_stark:
        print("\nComparative Analysis Across Runs:")
        print("-" * 40)
        
        # Time comparisons
        snark_total_times = []
        stark_total_times = []
        
        for run in range(args.runs):
            snark_compile = df.iloc[run]["snark_t_compile"]
            snark_setup = df.iloc[run]["snark_t_setup"]
            snark_proof = df.iloc[run]["snark_t_proof"]
            stark_build = df.iloc[run]["stark_t_build"]
            stark_setup = df.iloc[run]["stark_t_setup"]
            stark_proof = df.iloc[run]["stark_t_proof"]
            
            if not any(np.isnan([snark_compile, snark_setup, snark_proof])):
                snark_total_times.append(snark_compile + snark_setup + snark_proof)
            
            if not any(np.isnan([stark_build, stark_setup, stark_proof])):
                stark_total_times.append(stark_build + stark_setup + stark_proof)
        
        if snark_total_times and stark_total_times:
            speedups = [snark/stark for snark, stark in zip(snark_total_times, stark_total_times)]
            print(f"\nSpeedup (SNARK total / STARK total) per run:")
            print(f"  Values: {[f'{s:.1f}x' for s in speedups]}")
            stats = calculate_statistics(speedups)
            print(f"  Mean speedup: {stats['mean']:.1f}x ± {stats['std']:.1f}")
            print(f"  Range: [{stats['min']:.1f}x, {stats['max']:.1f}x]")
        
        # Proof size comparison
        snark_sizes = [df.iloc[i]["snark_proof_size"] for i in range(args.runs)]
        stark_sizes = [df.iloc[i]["stark_proof_size"] for i in range(args.runs)]
        
        if any(not np.isnan(s) for s in snark_sizes) and any(not np.isnan(s) for s in stark_sizes):
            size_ratios = []
            for snark_size, stark_size in zip(snark_sizes, stark_sizes):
                if not np.isnan(snark_size) and not np.isnan(stark_size) and snark_size > 0:
                    size_ratios.append(stark_size / snark_size)
            
            if size_ratios:
                print(f"\nProof size ratio (STARK / SNARK) per run:")
                print(f"  Values: {[f'{r:.1f}x' for r in size_ratios]}")
                stats = calculate_statistics(size_ratios)
                print(f"  Mean ratio: {stats['mean']:.1f}x ± {stats['std']:.1f}")
                print(f"  Range: [{stats['min']:.1f}x, {stats['max']:.1f}x]")
    
    # Recommendations based on consistency
    print("\nConsistency Assessment:")
    print("-" * 40)
    
    high_cv_metrics = []
    for col in df.columns:
        if col in ["batch", "run"]:
            continue
        values = [df.iloc[i][col] for i in range(args.runs)]
        stats = calculate_statistics(values)
        if stats['cv'] > 20:  # More than 20% coefficient of variation
            high_cv_metrics.append((col, stats['cv']))
    
    if high_cv_metrics:
        print("Metrics with high variability (CV > 20%):")
        for metric, cv in sorted(high_cv_metrics, key=lambda x: x[1], reverse=True):
            print(f"  {metric}: {cv:.1f}%")
        print("\nRecommendations:")
        print("- Consider running more iterations for these metrics")
        print("- Check for system load or other environmental factors")
        print("- Some variability in memory measurements is expected")
    else:
        print("All metrics show good consistency across runs!")
    
    print("\nAnalysis complete. Check the detailed results in:", output_file)