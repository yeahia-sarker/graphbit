//! Command handlers for GraphBit CLI
//!
//! This module contains handlers for all CLI commands, organized by functionality.

pub mod init;
pub mod run;
pub mod validate;
pub mod version;

// Re-export command handlers
pub use init::handle_init;
pub use run::handle_run;
pub use validate::handle_validate;
pub use version::handle_version;
