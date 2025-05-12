// src/debug.rs
//! Debug utilities for verifying STARK execution

use winterfell::math::fields::f128::BaseElement as Felt;
use winterfell::math::StarkField;
use winterfell::{Trace, TraceTable};
use std::fs::File;
use std::io::Write;

pub struct DebugTracer {
    enabled: bool,
    trace_data: Vec<(usize, String, Vec<f64>)>,
}

impl DebugTracer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            trace_data: Vec::new(),
        }
    }

    pub fn log_step(&mut self, step: usize, operation: &str, values: &[Felt]) {
        if !self.enabled { return; }
        
        let f64_values: Vec<f64> = values.iter()
            .map(|&f| f.as_int() as f64 / 1e6)
            .collect();
        
        // Print debug info before moving the data
        if step % 10 == 0 {
            println!("Step {}: {} - First few values: {:?}", 
                    step, operation, &f64_values[..f64_values.len().min(3)]);
        }
        
        // Now move the data
        self.trace_data.push((step, operation.to_string(), f64_values));
    }

    pub fn save_trace(&self, filename: &str) -> std::io::Result<()> {
        if !self.enabled { return Ok(()); }
        
        let mut file = File::create(filename)?;
        writeln!(file, "Step,Operation,Values")?;
        
        for (step, operation, values) in &self.trace_data {
            write!(file, "{},{}", step, operation)?;
            for value in values {
                write!(file, ",{:.6}", value)?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
}

// Function to verify trace consistency
pub fn verify_trace_transitions(trace: &TraceTable<Felt>, batch_size: usize) -> Result<(), String> {
    println!("\n=== Verifying Trace Transitions ===");
    
    let rows = trace.length();
    let cols = trace.width();
    let half_cols = cols / 2;
    
    // Check that masking/unmasking is consistent
    for row in 0..rows {
        for col in 0..half_cols {
            let masked = trace.get(col, row);
            let mask = trace.get(col + half_cols, row);
            let raw = masked - mask;
            
            // Check that raw values are reasonable (not overflow)
            if raw.as_int() > (u128::MAX / 2) {
                return Err(format!("Potential overflow at row {}, col {}: raw = {}", 
                                 row, col, raw.as_int()));
            }
        }
    }
    
    // Check that training steps only occur within batch_size
    let mut significant_changes = 0;
    for row in 1..rows.min(batch_size + 1) {
        let mut changes_in_row = 0;
        for col in 0..half_cols {
            let prev_masked = trace.get(col, row - 1);
            let prev_mask = trace.get(col + half_cols, row - 1);
            let prev_raw = prev_masked - prev_mask;
            
            let curr_masked = trace.get(col, row);
            let curr_mask = trace.get(col + half_cols, row);
            let curr_raw = curr_masked - curr_mask;
            
            if prev_raw != curr_raw {
                changes_in_row += 1;
            }
        }
        
        if changes_in_row > 0 {
            significant_changes += 1;
            println!("Row {}: {} columns changed", row, changes_in_row);
        }
    }
    
    println!("Found {} rows with significant changes (expected: {})", 
             significant_changes, batch_size);
    
    Ok(())
}

// Function to export trace data for external analysis
pub fn export_trace_csv(trace: &TraceTable<Felt>, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    write!(file, "Row")?;
    for col in 0..trace.width() {
        write!(file, ",Col_{}", col)?;
    }
    writeln!(file)?;
    
    // Write data
    for row in 0..trace.length() {
        write!(file, "{}", row)?;
        for col in 0..trace.width() {
            let value = trace.get(col, row);
            write!(file, ",{:.6}", value.as_int() as f64 / 1e6)?;
        }
        writeln!(file)?;
    }
    
    Ok(())
}

// Helper function to print trace analysis
pub fn analyze_trace(trace: &TraceTable<Felt>) {
    println!("Trace dimensions: {} rows x {} columns", trace.length(), trace.width());
    
    let half_width = trace.width() / 2;
    
    // Print initial state
    println!("\nInitial state (row 0):");
    for i in 0..half_width.min(10) {
        let masked = trace.get(i, 0);
        let mask = trace.get(i + half_width, 0);
        let raw = masked - mask;
        println!("  Column {}: masked={:.6}, mask={:.6}, raw={:.6}", 
                i, masked.as_int() as f64 / 1e6, mask.as_int() as f64 / 1e6, raw.as_int() as f64 / 1e6);
    }
    
    // Print final state
    println!("\nFinal state (row {}):", trace.length() - 1);
    for i in 0..half_width.min(10) {
        let masked = trace.get(i, trace.length() - 1);
        let mask = trace.get(i + half_width, trace.length() - 1);
        let raw = masked - mask;
        println!("  Column {}: masked={:.6}, mask={:.6}, raw={:.6}", 
                i, masked.as_int() as f64 / 1e6, mask.as_int() as f64 / 1e6, raw.as_int() as f64 / 1e6);
    }
    
    // Check for changes
    println!("\nState changes:");
    let mut changes = 0;
    for i in 0..half_width {
        let initial_masked = trace.get(i, 0);
        let initial_mask = trace.get(i + half_width, 0);
        let initial_raw = initial_masked - initial_mask;
        
        let final_masked = trace.get(i, trace.length() - 1);
        let final_mask = trace.get(i + half_width, trace.length() - 1);
        let final_raw = final_masked - final_mask;
        
        if initial_raw != final_raw {
            changes += 1;
            if changes <= 5 {
                println!("  Column {} changed: {:.6} -> {:.6} (diff: {:.6})", 
                        i, 
                        initial_raw.as_int() as f64 / 1e6, 
                        final_raw.as_int() as f64 / 1e6,
                        (final_raw - initial_raw).as_int() as f64 / 1e6);
            }
        }
    }
    println!("Total columns changed: {}/{}", changes, half_width);
}