// src-tauri/src/main.rs

//! Tauri backend — spawns a background thread that simulates EEG data
//! and emits `eeg-data` events to the frontend at ~30 Hz.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::Serialize;
use std::{thread, time::Duration};
use tauri::Emitter;

// Define the payload we will send to React
#[derive(Clone, Serialize)]
struct EEGPayload {
    valence: f32,
    arousal: f32,
    alpha: f32,
    beta: f32,
    raw_waves: Vec<f32>, // Mock raw wave data for the uPlot graph
}

/// Starts a mock EEG stream on a background thread.
/// Emits `eeg-data` events containing valence, arousal,
/// band power (alpha/beta), and raw wave samples.
#[tauri::command]
fn start_eeg_stream(app_handle: tauri::AppHandle) {
    println!("Starting EEG stream...");
    
    // Spawn a background thread so we don't block the UI
    thread::spawn(move || {
        let mut tick: f32 = 0.0; // Used to simulate changing metrics over time
        loop {
            // Simulate changing metrics
            let payload = EEGPayload {
                valence: (tick * 0.05).sin() * 0.5 + 0.5,
                arousal: (tick * 0.07).cos() * 0.5 + 0.5,
                alpha: (tick * 0.02).sin().abs(),
                beta: (tick * 0.03).cos().abs(),
                raw_waves: vec![(tick * 0.5).sin(), (tick * 0.6).cos(), (tick * 0.7).sin()], 
            };

            // Emit the event to the React frontend
            app_handle.emit("eeg-data", payload).unwrap();

            tick += 1.0;
            // Sleep for ~33ms to simulate 30 FPS
            thread::sleep(Duration::from_millis(33)); 
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![start_eeg_stream])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}