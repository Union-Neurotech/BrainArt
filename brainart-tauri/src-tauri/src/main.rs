// src-tauri/src/main.rs
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::Serialize;
use std::{thread, time::Duration};
use tauri::Manager;

// Define the payload we will send to React
#[derive(Clone, Serialize)]
struct EEGPayload {
    valence: f32,
    arousal: f32,
    alpha: f32,
    beta: f32,
    raw_waves: Vec<f32>, // Mock raw wave data for the uPlot graph
}

#[tauri::command]
fn start_eeg_stream(app_handle: tauri::AppHandle) {
    println!("Starting EEG stream...");
    
    // Spawn a background thread so we don't block the UI
    thread::spawn(move || {
        let mut tick = 0.0;
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
            app_handle.emit_all("eeg-data", payload).unwrap();

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