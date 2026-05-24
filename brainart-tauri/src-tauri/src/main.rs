// src-tauri/src/main.rs

//! Tauri backend — generates synthetic multi-band EEG, performs live FFT
//! band-power extraction, and emits `eeg-data` events at ~30 Hz.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;
use std::{collections::VecDeque, f32::consts::PI, thread, time::Duration};
use tauri::Emitter;

// ── Constants ─────────────────────────────────────────────────────────────────
const SAMPLE_RATE: f32 = 250.0;            // Hz — realistic EEG rate
const FFT_SIZE: usize = 256;               // ~1 s window at 250 Hz
const CHANNELS: usize = 3;
const EMIT_MS: u64 = 33;                   // ~30 FPS emit rate
// Samples to generate per emit frame  (250 Hz × 0.033 s ≈ 8)
const SPF: usize = ((SAMPLE_RATE / 1000.0) * EMIT_MS as f32) as usize;

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

// ── FFT helpers ───────────────────────────────────────────────────────────────

/// Sum power in [lo_hz, hi_hz) from a one-sided power spectrum.
fn band_power(spectrum: &[f32], lo_hz: f32, hi_hz: f32) -> f32 {
    let bin = |hz: f32| (hz * FFT_SIZE as f32 / SAMPLE_RATE).round() as usize;
    let lo = bin(lo_hz).max(1);
    let hi = bin(hi_hz).min(FFT_SIZE / 2);
    spectrum[lo..hi].iter().sum()
}

/// Compute normalised (relative) band powers from a rolling sample buffer.
/// Returns (delta, theta, alpha, beta, gamma) each in [0, 1].
fn compute_bands(buf: &VecDeque<f32>) -> (f32, f32, f32, f32, f32) {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Fill from the most recent FFT_SIZE samples; zero-pad if buffer is short
    let start = buf.len().saturating_sub(FFT_SIZE);
    let mut input: Vec<Complex<f32>> = (0..FFT_SIZE)
        .map(|i| {
            let sample = buf.get(start + i).copied().unwrap_or(0.0);
            // Hann window to reduce spectral leakage
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos());
            Complex { re: sample * w, im: 0.0 }
        })
        .collect();

    fft.process(&mut input);

    // One-sided power spectrum
    let power: Vec<f32> = input[..FFT_SIZE / 2]
        .iter()
        .map(|c| c.norm_sqr() / FFT_SIZE as f32)
        .collect();

    let d = band_power(&power,  0.5,  4.0);
    let t = band_power(&power,  4.0,  8.0);
    let a = band_power(&power,  8.0, 13.0);
    let b = band_power(&power, 13.0, 30.0);
    let g = band_power(&power, 30.0, SAMPLE_RATE / 2.0);

    // Relative power: normalise so bands sum to 1
    let total = d + t + a + b + g + 1e-12;
    (d / total, t / total, a / total, b / total, g / total)
}

// ── Tauri command ─────────────────────────────────────────────────────────────

/// Connect to the specified board and start streaming EEG data.
/// Currently all board types run the synthetic signal generator.
#[tauri::command]
fn connect_board(board_id: String, app_handle: tauri::AppHandle) {
    println!("Connecting to board: {}", board_id);

    thread::spawn(move || {
        let mut sample_idx: u64 = 0;
        let mut buffers: Vec<VecDeque<f32>> =
            vec![VecDeque::with_capacity(FFT_SIZE + SPF + 1); CHANNELS];

        loop {
            // Generate SPF new samples per emit tick
            let mut ch_last = [0.0_f32; CHANNELS];

            for _ in 0..SPF {
                let t = sample_idx as f32 / SAMPLE_RATE;

                // Slowly-modulated amplitudes give each band realistic drift
                let d_amp = 0.40 + 0.30 * (t * 0.07).sin();
                let t_amp = 0.20 + 0.18 * (t * 0.11).cos();
                let a_amp = 0.35 + 0.25 * (t * 0.05).sin();
                let b_amp = 0.15 + 0.10 * (t * 0.13).cos();
                let g_amp = 0.08 + 0.06 * (t * 0.09).sin();

                for ch in 0..CHANNELS {
                    let phi = ch as f32 * 0.4; // phase offset per channel
                    let s = d_amp * (2.0 * PI *  2.0 * t + phi).sin()
                          + t_amp * (2.0 * PI *  6.0 * t + phi).sin()
                          + a_amp * (2.0 * PI * 10.0 * t + phi).sin()
                          + b_amp * (2.0 * PI * 20.0 * t + phi).sin()
                          + g_amp * (2.0 * PI * 40.0 * t + phi).sin();

                    buffers[ch].push_back(s);
                    // Keep buffer bounded
                    if buffers[ch].len() > FFT_SIZE + SPF + 1 {
                        buffers[ch].pop_front();
                    }
                    ch_last[ch] = s;
                }
                sample_idx += 1;
            }

            // FFT band powers from channel 0 (primary EEG channel)
            let (delta, theta, alpha, beta, gamma) = if buffers[0].len() >= FFT_SIZE {
                compute_bands(&buffers[0])
            } else {
                (0.2, 0.2, 0.2, 0.2, 0.2) // flat until buffer fills (~1 s)
            };

            // Simple neurofeedback-style mappings
            let valence       = (alpha - beta).clamp(-1.0, 1.0);
            let arousal       = ((beta + gamma) - (alpha + theta)).clamp(-1.0, 1.0);
            let mindfulness   = (alpha + theta).min(1.0);
            let concentration = beta.min(1.0) * 2.0_f32.min(1.0);
            let relaxation    = (alpha + theta * 0.5).min(1.0);

            let payload = EEGPayload {
                valence,
                arousal,
                alpha,
                beta,
                theta,
                delta,
                gamma,
                mindfulness,
                concentration: concentration.clamp(0.0, 1.0),
                relaxation,
                // One sample per channel for the rolling uPlot display
                raw_waves: ch_last.to_vec(),
            };

            app_handle.emit("eeg-data", payload).unwrap();
            thread::sleep(Duration::from_millis(EMIT_MS));
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![connect_board])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

