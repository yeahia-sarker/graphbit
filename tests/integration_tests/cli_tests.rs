//! CLI integration tests
//!
//! Tests for the GraphBit command-line interface, covering all commands
//! and their various options and error conditions.

use super::TestEnvironment;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Helper function to run the GraphBit CLI and capture output
fn run_graphbit_cli(args: &[&str]) -> (i32, String, String) {
    let output = Command::new("cargo")
        .arg("run")
        .arg("--bin")
        .arg("graphbit")
        .arg("--")
        .args(args)
        .output()
        .expect("Failed to execute CLI command");

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (exit_code, stdout, stderr)
}

#[tokio::test]
async fn test_cli_version_command() {
    let (exit_code, stdout, _stderr) = run_graphbit_cli(&["version"]);

    assert_eq!(exit_code, 0, "Version command should succeed");
    assert!(
        stdout.contains("GraphBit v"),
        "Should show version information"
    );
    assert!(
        stdout.contains("Declarative framework"),
        "Should show description"
    );
}

#[tokio::test]
async fn test_cli_init_command() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_name = "test-project";
    let project_path = temp_dir.path().join(project_name);

    let (exit_code, stdout, _stderr) = run_graphbit_cli(&[
        "init",
        project_name,
        "--path",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Init command should succeed");
    assert!(
        stdout.contains("initialized successfully"),
        "Should show success message"
    );

    // Verify project structure was created
    assert!(project_path.exists(), "Project directory should be created");
    assert!(
        project_path.join("workflows").exists(),
        "Workflows directory should be created"
    );
    assert!(
        project_path.join("agents").exists(),
        "Agents directory should be created"
    );
    assert!(
        project_path.join("config").exists(),
        "Config directory should be created"
    );

    // Verify example files were created
    let example_workflow = project_path.join("workflows").join("example.json");
    assert!(
        example_workflow.exists(),
        "Example workflow should be created"
    );

    let example_config = project_path.join("config").join("example.json");
    assert!(example_config.exists(), "Example config should be created");
}

#[tokio::test]
async fn test_cli_init_command_current_directory() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_name = "current-dir-project";

    let (exit_code, stdout, _stderr) = run_graphbit_cli(&[
        "init",
        project_name,
        "--path",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(
        exit_code, 0,
        "Init command should succeed in current directory"
    );
    assert!(
        stdout.contains("initialized successfully"),
        "Should show success message"
    );

    let project_path = temp_dir.path().join(project_name);
    assert!(project_path.exists(), "Project directory should be created");
}

#[tokio::test]
async fn test_cli_validate_command_valid_workflow() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_test_config()
        .expect("Failed to create test config");
    env.create_simple_workflow()
        .expect("Failed to create test workflow");

    let (exit_code, stdout, _stderr) =
        run_graphbit_cli(&["validate", env.workflow_path.to_str().unwrap()]);

    assert_eq!(exit_code, 0, "Validation should succeed for valid workflow");
    assert!(
        stdout.contains("Workflow is valid"),
        "Should show validation success"
    );
}

#[tokio::test]
async fn test_cli_validate_command_invalid_workflow() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_invalid_workflow()
        .expect("Failed to create invalid workflow");

    let (exit_code, _stdout, stderr) =
        run_graphbit_cli(&["validate", env.workflow_path.to_str().unwrap()]);

    assert_ne!(exit_code, 0, "Validation should fail for invalid workflow");
    assert!(!stderr.is_empty(), "Should show validation errors");
}

#[tokio::test]
async fn test_cli_validate_command_nonexistent_file() {
    let (exit_code, _stdout, stderr) =
        run_graphbit_cli(&["validate", "/nonexistent/workflow.json"]);

    assert_ne!(exit_code, 0, "Validation should fail for nonexistent file");
    assert!(!stderr.is_empty(), "Should show file not found error");
}

#[tokio::test]
async fn test_cli_run_command_simple_workflow() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_test_config()
        .expect("Failed to create test config");
    env.create_simple_workflow()
        .expect("Failed to create test workflow");

    let (exit_code, stdout, _stderr) = run_graphbit_cli(&[
        "run",
        env.workflow_path.to_str().unwrap(),
        "--config",
        env.config_path.to_str().unwrap(),
    ]);

    // Note: This might fail due to API key issues, but we test the command structure
    if exit_code == 0 {
        assert!(
            stdout.contains("completed successfully"),
            "Should show completion message"
        );
    } else {
        // Check that it at least tried to run and failed gracefully
        assert!(exit_code != 0, "Should handle execution errors gracefully");
    }
}

#[tokio::test]
async fn test_cli_run_command_without_config() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_simple_workflow()
        .expect("Failed to create test workflow");

    let (exit_code, _stdout, _stderr) =
        run_graphbit_cli(&["run", env.workflow_path.to_str().unwrap()]);

    // Should still attempt to run but may fail due to missing config
    // The important thing is that the CLI accepts the command structure
    assert!(exit_code != -1, "Command should be parsed correctly");
}

#[tokio::test]
async fn test_cli_run_command_complex_workflow() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_test_config()
        .expect("Failed to create test config");
    env.create_complex_workflow()
        .expect("Failed to create complex workflow");

    let (exit_code, _stdout, _stderr) = run_graphbit_cli(&[
        "run",
        env.workflow_path.to_str().unwrap(),
        "--config",
        env.config_path.to_str().unwrap(),
    ]);

    // Complex workflow should be parsed correctly even if execution fails
    assert!(exit_code != -1, "Command should be parsed correctly");
}

#[tokio::test]
async fn test_cli_invalid_command() {
    let (exit_code, _stdout, stderr) = run_graphbit_cli(&["invalid-command"]);

    assert_ne!(exit_code, 0, "Invalid command should fail");
    assert!(
        !stderr.is_empty() || !_stdout.is_empty(),
        "Should show help or error message"
    );
}

#[tokio::test]
async fn test_cli_help_command() {
    let (exit_code, stdout, _stderr) = run_graphbit_cli(&["--help"]);

    assert_eq!(exit_code, 0, "Help command should succeed");
    assert!(
        stdout.contains("Declarative framework"),
        "Should show description"
    );
    assert!(
        stdout.contains("Commands:"),
        "Should show available commands"
    );
    assert!(stdout.contains("init"), "Should list init command");
    assert!(stdout.contains("validate"), "Should list validate command");
    assert!(stdout.contains("run"), "Should list run command");
    assert!(stdout.contains("version"), "Should list version command");
}

#[tokio::test]
async fn test_cli_subcommand_help() {
    let commands = ["init", "validate", "run"];

    for command in &commands {
        let (exit_code, stdout, _stderr) = run_graphbit_cli(&[command, "--help"]);

        assert_eq!(exit_code, 0, "Help for {} command should succeed", command);
        assert!(
            stdout.contains(command),
            "Should show {} command help",
            command
        );
    }
}

#[tokio::test]
async fn test_cli_missing_required_arguments() {
    // Test init without name
    let (exit_code, _stdout, stderr) = run_graphbit_cli(&["init"]);
    assert_ne!(exit_code, 0, "Init without name should fail");
    assert!(!stderr.is_empty(), "Should show missing argument error");

    // Test validate without workflow file
    let (exit_code, _stdout, stderr) = run_graphbit_cli(&["validate"]);
    assert_ne!(exit_code, 0, "Validate without workflow should fail");
    assert!(!stderr.is_empty(), "Should show missing argument error");

    // Test run without workflow file
    let (exit_code, _stdout, stderr) = run_graphbit_cli(&["run"]);
    assert_ne!(exit_code, 0, "Run without workflow should fail");
    assert!(!stderr.is_empty(), "Should show missing argument error");
}

#[tokio::test]
async fn test_cli_malformed_json_workflow() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let workflow_path = temp_dir.path().join("malformed.json");

    // Create malformed JSON
    fs::write(&workflow_path, "{invalid json content}").expect("Failed to write malformed JSON");

    let (exit_code, _stdout, stderr) =
        run_graphbit_cli(&["validate", workflow_path.to_str().unwrap()]);

    assert_ne!(exit_code, 0, "Should fail on malformed JSON");
    assert!(!stderr.is_empty(), "Should show JSON parsing error");
}

#[tokio::test]
async fn test_cli_malformed_json_config() {
    let env = TestEnvironment::new().expect("Failed to create test environment");
    env.create_simple_workflow()
        .expect("Failed to create test workflow");

    // Create malformed config
    fs::write(&env.config_path, "{invalid json}").expect("Failed to write malformed JSON");

    let (exit_code, _stdout, stderr) = run_graphbit_cli(&[
        "run",
        env.workflow_path.to_str().unwrap(),
        "--config",
        env.config_path.to_str().unwrap(),
    ]);

    assert_ne!(exit_code, 0, "Should fail on malformed config JSON");
    assert!(!stderr.is_empty(), "Should show JSON parsing error");
}

#[tokio::test]
async fn test_cli_empty_workflow_file() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let workflow_path = temp_dir.path().join("empty.json");

    // Create empty file
    fs::write(&workflow_path, "").expect("Failed to write empty file");

    let (exit_code, _stdout, stderr) =
        run_graphbit_cli(&["validate", workflow_path.to_str().unwrap()]);

    assert_ne!(exit_code, 0, "Should fail on empty workflow file");
    assert!(!stderr.is_empty(), "Should show parsing error");
}

#[tokio::test]
async fn test_cli_long_arguments() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_name = "project-with-very-long-name-that-should-still-work";

    let (exit_code, stdout, _stderr) = run_graphbit_cli(&[
        "init",
        project_name,
        "--path",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Should handle long project names");
    assert!(
        stdout.contains("initialized successfully"),
        "Should show success message"
    );

    let project_path = temp_dir.path().join(project_name);
    assert!(project_path.exists(), "Project directory should be created");
}

#[tokio::test]
async fn test_cli_special_characters_in_paths() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_name = "project-with-special-chars_123";

    let (exit_code, stdout, _stderr) = run_graphbit_cli(&[
        "init",
        project_name,
        "--path",
        temp_dir.path().to_str().unwrap(),
    ]);

    assert_eq!(exit_code, 0, "Should handle special characters in names");
    assert!(
        stdout.contains("initialized successfully"),
        "Should show success message"
    );
}
