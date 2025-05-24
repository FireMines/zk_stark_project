#!/usr/bin/env python3
"""
Simple batch size automation script with progress tracking
Runs analyze.py exactly as you would manually, just automated with timing info
"""

import subprocess
import sys
import time
import datetime

# Configuration - change these as needed
BATCH_SIZES = [1, 10, 15, 20, 25, 30, 35, 40, 45, 50]
RUNS = 11

def log_message(message):
    """Print timestamped message"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def format_duration(seconds):
    """Format duration in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def estimate_total_time(batch_sizes, runs):
    """Estimate total runtime based on batch sizes"""
    # Rough estimates based on typical performance (seconds per run)
    snark_times = {1: 10, 10: 128, 20: 292, 30: 558, 40: 1012}
    stark_times = {1: 7, 10: 9, 20: 12, 30: 12, 40: 16}
    
    total_estimate = 0
    for batch in batch_sizes:
        total_estimate += (snark_times.get(batch, batch * 25) + stark_times.get(batch, batch * 2)) * runs
    
    return total_estimate

def main():
    log_message("=== Simple Batch Automation ===")
    log_message(f"Batch sizes: {BATCH_SIZES}")
    log_message(f"Runs per batch: {RUNS}")
    
    # Estimate total time
    estimated_time = estimate_total_time(BATCH_SIZES, RUNS)
    log_message(f"Estimated total runtime: {format_duration(estimated_time)}")
    log_message("=" * 50)
    
    overall_start = time.time()
    
    for i, batch_size in enumerate(BATCH_SIZES, 1):
        log_message(f"=== Batch {i}/{len(BATCH_SIZES)}: Size {batch_size} ===")
        log_message(f"Starting benchmark for batch size {batch_size} ({RUNS} runs)")
        
        # Run exactly the same command as you would manually
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "analyze.py",
            "--runs", str(RUNS),
            "--batch-size", str(batch_size)
        ]
        
        log_message(f"Running: {' '.join(cmd)}")
        
        # Run the command and wait for completion
        start_time = time.time()
        result = subprocess.run(cmd)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            log_message(f"✓ Batch size {batch_size} completed successfully in {format_duration(duration)}")
        else:
            log_message(f"✗ Batch size {batch_size} failed with return code {result.returncode}")
            log_message("Continuing with next batch...")
        
        # Progress tracking
        completed = i
        remaining = len(BATCH_SIZES) - completed
        elapsed_total = time.time() - overall_start
        
        if completed > 0:
            avg_time_per_batch = elapsed_total / completed
            estimated_remaining = avg_time_per_batch * remaining
            
            log_message(f"Progress: {completed}/{len(BATCH_SIZES)} batches completed")
            log_message(f"Elapsed total: {format_duration(elapsed_total)}")
            if remaining > 0:
                log_message(f"Estimated remaining: {format_duration(estimated_remaining)}")
        
        log_message("-" * 40)
    
    # Final summary
    total_time = time.time() - overall_start
    log_message("=== AUTOMATION COMPLETE ===")
    log_message(f"Total runtime: {format_duration(total_time)}")
    log_message(f"Completed all {len(BATCH_SIZES)} batch sizes")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\nAutomation interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_message(f"Automation failed with error: {e}")
        sys.exit(1)