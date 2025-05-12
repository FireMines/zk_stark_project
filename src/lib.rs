// src/lib.rs
//! Library root for zk_stark_project

pub mod helper;
pub mod signed;
pub mod debug;

// Make sure these modules are properly exposed
pub mod training {
    pub mod air;
    pub mod prover;
}

pub mod aggregation {
    pub mod air;
    pub mod prover;
}

// Re-export commonly used items
pub use helper::*;
pub use signed::*;