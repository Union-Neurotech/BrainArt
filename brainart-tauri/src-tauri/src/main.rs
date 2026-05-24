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

/// Full EEG payload — matches the EegState interface in App.tsx
#[derive(Clone, Serialize)]
struct EEGPayload {
    valence: f32,
    arousal: f32,
    alpha: f32,
    beta: f32,
    theta: f32,
    delta: f32,
    gamma: f32,
    mindfulness: f32,
    concentration: f32,
    relaxation: f32,
    raw_waves: Vec<f32>,
}

/// Connect to the specified board and start streaming EEG data.
/// Currently all board types run the synthetic mock stream.
#[tauri::command]
fn connect_board(board_id: String, app_handle: tauri::AppHandle) {
    println!("Connecting to board: {}", board_id);

    thread::spawn(move || {
        let mut tick: f32 = 0.0;
        loop {
            let payload = EEGPayload {
                valence:       (tick * 0.05).sin(),
                arousal:       (tick * 0.07).cos(),
                alpha:         (tick * 0.02).sin().abs(),
                beta:          (tick * 0.03).cos().abs(),
                theta:         (tick * 0.04).sin().abs(),
                delta:         (tick * 0.015).cos().abs(),
                gamma:         (tick * 0.06).sin().abs(),
                mindfulness:   (tick * 0.01).sin() * 0.5 + 0.5,
                concentration: (tick * 0.013).cos() * 0.5 + 0.5,
                relaxation:    (tick * 0.008).sin() * 0.5 + 0.5,
                raw_waves: vec![
                    (tick * 0.5).sin(),
                    (tick * 0.6).cos(),
                    (tick * 0.7).sin(),
                ],
            };

            app_handle.emit("eeg-data", payload).unwrap();
            tick += 1.0;
            thread::sleep(Duration::from_millis(33));
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![connect_board])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
