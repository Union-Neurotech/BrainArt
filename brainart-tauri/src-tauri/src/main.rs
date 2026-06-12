// src-tauri/src/main.rs

//! Tauri backend for the live BrainArt app.
//!
//! Instead of simulating EEG in Rust, this launches the existing Python
//! inference server (`emotion_pipeline/brainart_muse_server.py`) as a child
//! process. That server owns the Muse via BrainFlow, runs the valence/arousal
//! model + calibration, and broadcasts JSON over `ws://localhost:8765`. The
//! React frontend connects to that WebSocket directly (see App.tsx).
//!
//! Board selection is mapped to the server's CLI flags:
//!   synthetic -> --sim         (no hardware, synthetic stream)
//!   muse_2    -> --board muse_2
//!   muse_2016 -> --board muse_2016
//!   muse_s    -> --board muse_s

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, State, WindowEvent};

/// Holds the running Python server process so we can stop/replace it.
#[derive(Default)]
struct ServerProc(Mutex<Option<Child>>);

/// Locate `emotion_pipeline/` relative to this crate.
/// CARGO_MANIFEST_DIR is `.../brainart-tauri/src-tauri` at build time, so the
/// repo root is two levels up. Overridable at runtime with BRAINART_PIPELINE.
fn pipeline_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BRAINART_PIPELINE") {
        return PathBuf::from(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent() // .../brainart-tauri
        .and_then(|p| p.parent()) // repo root
        .map(|repo| repo.join("emotion_pipeline"))
        .unwrap_or_else(|| PathBuf::from("emotion_pipeline"))
}

/// Python executable: BRAINART_PYTHON, else `python` (Win) / `python3` (unix).
fn python_exe() -> String {
    std::env::var("BRAINART_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) { "python".into() } else { "python3".into() }
    })
}

fn kill_existing(state: &State<ServerProc>) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Launch the Python inference server for the selected board.
#[tauri::command]
fn start_server(board_id: String, state: State<ServerProc>) -> Result<String, String> {
    let args: Vec<&str> = match board_id.as_str() {
        "synthetic" => vec!["--sim"],
        "muse_2" => vec!["--board", "muse_2"],
        "muse_2016" => vec!["--board", "muse_2016"],
        "muse_s" => vec!["--board", "muse_s"],
        other => return Err(format!("unsupported board '{other}'")),
    };

    let dir = pipeline_dir();
    let script = dir.join("brainart_muse_server.py");
    if !script.exists() {
        return Err(format!(
            "server script not found at {} (set BRAINART_PIPELINE to override)",
            script.display()
        ));
    }

    kill_existing(&state);

    let child = Command::new(python_exe())
        .arg("brainart_muse_server.py")
        .args(&args)
        .current_dir(&dir)
        .spawn()
        .map_err(|e| format!("failed to launch Python server ({}): {e}", python_exe()))?;

    *state.0.lock().map_err(|e| e.to_string())? = Some(child);
    Ok(format!("server started for board '{board_id}'"))
}

/// Stop the Python inference server if running.
#[tauri::command]
fn stop_server(state: State<ServerProc>) -> Result<(), String> {
    kill_existing(&state);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(ServerProc::default())
        .invoke_handler(tauri::generate_handler![start_server, stop_server])
        .on_window_event(|window, event| {
            // Make sure the Python child dies with the window.
            if let WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<ServerProc>() {
                    kill_existing(&state);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
