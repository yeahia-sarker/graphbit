//! Project initialization command handler
//!
//! This module handles the `init` command for creating new GraphBit projects.

use crate::cli::project::create_project_structure;
use crate::cli::utils::get_ollama_model_recommendations;
use graphbit_core::GraphBitResult;
use std::path::PathBuf;

/// Handle the init command
pub async fn handle_init(name: String, path: Option<PathBuf>) -> GraphBitResult<()> {
    let project_path = path.unwrap_or_else(|| std::env::current_dir().unwrap());

    println!(
        "Initializing GraphBit project '{}' in {:?}",
        name, project_path
    );

    // Create project structure
    create_project_structure(&project_path, &name)?;

    println!("✅ Project '{}' initialized successfully!", name);
    println!();
    println!("Next steps:");
    println!("  1. cd {}", name);
    println!("  2. cp .env.example .env");
    println!("  3. Edit .env with your API keys");
    println!("  4. graphbit validate workflows/example.json");
    println!("  5. graphbit run workflows/example.json --config config/example.json");

    println!();
    println!("💡 Popular Ollama models for local development:");
    let recommendations = get_ollama_model_recommendations();
    for (model, description) in recommendations {
        println!("  • {} - {}", model, description);
    }
    println!(
        "  To use Ollama: Start with 'ollama serve' and pull a model like 'ollama pull llama3.1'"
    );

    Ok(())
}
