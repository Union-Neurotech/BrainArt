// src-tauri/src/main.rs

//! Tauri backend — EEG streaming with two code paths:
//!
//! * **Synthetic board**: built-in multi-band signal generator + rustfft.
//!   Works out of the box with no extra dependencies.
//!
//! * **Hardware boards** (Muse 2016/2, OpenBCI Cyton, …): routed through the
//!   BrainFlow Rust binding when compiled with `--features brainflow-hardware`.
//!   Requires the BrainFlow SDK to be installed locally first — see Cargo.toml.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, f32::consts::PI, thread, time::Duration};
use tauri::Emitter;

// ── Constants ─────────────────────────────────────────────────────────────────
const SAMPLE_RATE: f32 = 250.0;
const FFT_SIZE: usize = 256;
const CHANNELS: usize = 3;
const EMIT_MS: u64 = 33; // ~30 FPS
const SPF: usize = ((SAMPLE_RATE / 1000.0) * EMIT_MS as f32) as usize;

// ── Shared payload ────────────────────────────────────────────────────────────

/// Full EEG payload — mirrors the `EegState` interface in App.tsx.
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

/// Parameters forwarded from the frontend connect screen.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConnectRequest {
    board_id: String,
    /// Serial port for Cyton (e.g. "COM3") or Muse dongle.
    serial_port: Option<String>,
    /// Bluetooth MAC address for Muse boards (e.g. "XX:XX:XX:XX:XX:XX").
    mac_address: Option<String>,
    /// IP address for WiFi-connected boards (e.g. OpenBCI WiFi Shield).
    ip_address: Option<String>,
    /// IP port for WiFi-connected boards.
    ip_port: Option<i32>,
    /// Board connection timeout in seconds.
    timeout: Option<i32>,
}

// ── Neurofeedback mappings (shared by both paths) ─────────────────────────────

fn derive_payload(
    delta: f32, theta: f32, alpha: f32, beta: f32, gamma: f32,
    raw_waves: Vec<f32>,
) -> EEGPayload {
    let valence       = (alpha - beta).clamp(-1.0, 1.0);
    let arousal       = ((beta + gamma) - (alpha + theta)).clamp(-1.0, 1.0);
    let mindfulness   = (alpha + theta).min(1.0);
    let concentration = (beta * 2.0).clamp(0.0, 1.0);
    let relaxation    = (alpha + theta * 0.5).min(1.0);
    EEGPayload { valence, arousal, alpha, beta, theta, delta, gamma,
                 mindfulness, concentration, relaxation, raw_waves }
}

// ── Synthetic path (rustfft) ──────────────────────────────────────────────────

fn band_power(spectrum: &[f32], lo_hz: f32, hi_hz: f32) -> f32 {
    let bin = |hz: f32| (hz * FFT_SIZE as f32 / SAMPLE_RATE).round() as usize;
    spectrum[bin(lo_hz).max(1)..bin(hi_hz).min(FFT_SIZE / 2)].iter().sum()
}

fn compute_bands(buf: &VecDeque<f32>) -> (f32, f32, f32, f32, f32) {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let start = buf.len().saturating_sub(FFT_SIZE);
    let mut input: Vec<Complex<f32>> = (0..FFT_SIZE)
        .map(|i| {
            let s = buf.get(start + i).copied().unwrap_or(0.0);
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos());
            Complex { re: s * w, im: 0.0 }
        })
        .collect();
    fft.process(&mut input);
    let power: Vec<f32> = input[..FFT_SIZE / 2]
        .iter().map(|c| c.norm_sqr() / FFT_SIZE as f32).collect();
    let d = band_power(&power,  0.5,  4.0);
    let t = band_power(&power,  4.0,  8.0);
    let a = band_power(&power,  8.0, 13.0);
    let b = band_power(&power, 13.0, 30.0);
    let g = band_power(&power, 30.0, SAMPLE_RATE / 2.0);
    let total = d + t + a + b + g + 1e-12;
    (d / total, t / total, a / total, b / total, g / total)
}

fn run_synthetic_stream(app_handle: tauri::AppHandle) {
    thread::spawn(move || {
        let mut sample_idx: u64 = 0;
        let mut buffers: Vec<VecDeque<f32>> =
            vec![VecDeque::with_capacity(FFT_SIZE + SPF + 1); CHANNELS];

        loop {
            let mut ch_last = [0.0_f32; CHANNELS];

            for _ in 0..SPF {
                let t = sample_idx as f32 / SAMPLE_RATE;
                let d_amp = 0.40 + 0.30 * (t * 0.07).sin();
                let t_amp = 0.20 + 0.18 * (t * 0.11).cos();
                let a_amp = 0.35 + 0.25 * (t * 0.05).sin();
                let b_amp = 0.15 + 0.10 * (t * 0.13).cos();
                let g_amp = 0.08 + 0.06 * (t * 0.09).sin();

                for ch in 0..CHANNELS {
                    let phi = ch as f32 * 0.4;
                    let s = d_amp * (2.0 * PI *  2.0 * t + phi).sin()
                          + t_amp * (2.0 * PI *  6.0 * t + phi).sin()
                          + a_amp * (2.0 * PI * 10.0 * t + phi).sin()
                          + b_amp * (2.0 * PI * 20.0 * t + phi).sin()
                          + g_amp * (2.0 * PI * 40.0 * t + phi).sin();
                    buffers[ch].push_back(s);
                    if buffers[ch].len() > FFT_SIZE + SPF + 1 { buffers[ch].pop_front(); }
                    ch_last[ch] = s;
                }
                sample_idx += 1;
            }

            let (delta, theta, alpha, beta, gamma) = if buffers[0].len() >= FFT_SIZE {
                compute_bands(&buffers[0])
            } else {
                (0.2, 0.2, 0.2, 0.2, 0.2)
            };

            let payload = derive_payload(delta, theta, alpha, beta, gamma,
                                         ch_last.to_vec());
            app_handle.emit("eeg-data", payload).unwrap();
            thread::sleep(Duration::from_millis(EMIT_MS));
        }
    });
}

// ── BrainFlow hardware path ───────────────────────────────────────────────────
//
// All code in this block is compiled only when the `brainflow-hardware` feature
// is enabled.  The feature is disabled by default; see Cargo.toml for setup
// instructions.

#[cfg(feature = "brainflow-hardware")]
mod hardware {
    use brainflow::{
        board_shim::{self, BoardShim},
        brainflow_input_params::BrainFlowInputParamsBuilder,
        data_filter,
        BoardIds, BrainFlowPresets,
    };
    use std::thread;
    use std::time::Duration;
    use tauri::Emitter;

    use super::{derive_payload, ConnectRequest, EMIT_MS};

    /// Map the string board ID sent from the frontend to a BrainFlow `BoardIds`
    /// enum variant.
    ///
    /// BrainFlow board IDs:
    ///   SYNTHETIC_BOARD   = -1
    ///   CYTON_BOARD       =  0
    ///   MUSE_2016_BOARD   = 41
    ///   MUSE_S_BLED_BOARD = 38  (Muse 2 / Muse S via Bluetooth LE)
    fn str_to_board_id(s: &str) -> BoardIds {
        match s {
            "muse_2016" => BoardIds::Muse2016Board,
            "muse_2"    => BoardIds::MuseSBledBoard,
            "cyton"     => BoardIds::CytonBoard,
            _           => BoardIds::SyntheticBoard,
        }
    }

    pub fn run_hardware_stream(req: ConnectRequest, app_handle: tauri::AppHandle) {
        thread::spawn(move || {
            let board_id = str_to_board_id(&req.board_id);

            // Build connection params from whatever the frontend provided.
            let mut builder = BrainFlowInputParamsBuilder::default();
            if let Some(port) = req.serial_port.filter(|s| !s.is_empty()) {
                builder = builder.serial_port(port);
            }
            if let Some(mac) = req.mac_address.filter(|s| !s.is_empty()) {
                builder = builder.mac_address(mac);
            }
            if let Some(ip) = req.ip_address.filter(|s| !s.is_empty()) {
                builder = builder.ip_address(ip);
            }
            if let Some(port) = req.ip_port {
                builder = builder.ip_port(port as usize);
            }
            if let Some(t) = req.timeout {
                builder = builder.timeout(t as usize);
            }
            let params = builder.build();

            let board = match BoardShim::new(board_id, params) {
                Ok(b)  => b,
                Err(e) => { eprintln!("BrainFlow: failed to create BoardShim: {e}"); return; }
            };
            if let Err(e) = board.prepare_session() {
                eprintln!("BrainFlow: prepare_session failed: {e}"); return;
            }
            if let Err(e) = board.start_stream(45_000, "") {
                eprintln!("BrainFlow: start_stream failed: {e}");
                let _ = board.release_session();
                return;
            }

            let eeg_channels = match board_shim::get_eeg_channels(
                board_id, BrainFlowPresets::DefaultPreset)
            {
                Ok(ch) => ch,
                Err(e) => { eprintln!("BrainFlow: get_eeg_channels failed: {e}"); return; }
            };

            let sampling_rate = match board_shim::get_sampling_rate(
                board_id, BrainFlowPresets::DefaultPreset)
            {
                Ok(r)  => r as usize,
                Err(e) => { eprintln!("BrainFlow: get_sampling_rate failed: {e}"); return; }
            };

            loop {
                thread::sleep(Duration::from_millis(EMIT_MS));

                let data = match board.get_current_board_data(
                    sampling_rate, BrainFlowPresets::DefaultPreset)
                {
                    Ok(d)  => d,
                    Err(e) => { eprintln!("BrainFlow: get_current_board_data: {e}"); break; }
                };

                let n_samples = data.ncols();
                if n_samples < 2 { continue; }

                // Grab the latest raw sample from the first 3 EEG channels
                // (must happen before data is moved into get_avg_band_powers).
                let raw_waves: Vec<f32> = eeg_channels.iter().take(3)
                    .map(|&ch| data[[ch, n_samples - 1]] as f32)
                    .collect();

                // BrainFlow's built-in DSP: returns [delta, theta, alpha, beta, gamma]
                let avg_bands = match data_filter::get_avg_band_powers(
                    data, eeg_channels.clone(), sampling_rate, true)
                {
                    Ok((bands, _)) => bands,
                    Err(_) => continue,
                };

                let delta = avg_bands[0] as f32;
                let theta = avg_bands[1] as f32;
                let alpha = avg_bands[2] as f32;
                let beta  = avg_bands[3] as f32;
                let gamma = avg_bands[4] as f32;

                let payload = derive_payload(delta, theta, alpha, beta, gamma, raw_waves);
                if app_handle.emit("eeg-data", payload).is_err() { break; }
            }

            let _ = board.stop_stream();
            let _ = board.release_session();
        });
    }
}

// ── Tauri command ─────────────────────────────────────────────────────────────

#[tauri::command]
fn connect_board(request: ConnectRequest, app_handle: tauri::AppHandle) {
    println!("Connecting to board: {} (serial={:?}, mac={:?})",
             request.board_id, request.serial_port, request.mac_address);

    // When brainflow-hardware is enabled, ALL board types — including synthetic —
    // go through BrainFlow's own BoardShim (BoardIds::SyntheticBoard for "synthetic").
    #[cfg(feature = "brainflow-hardware")]
    hardware::run_hardware_stream(request, app_handle);

    // Fallback: built-in rustfft synthetic generator when the feature is disabled.
    #[cfg(not(feature = "brainflow-hardware"))]
    run_synthetic_stream(app_handle);
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![connect_board])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

