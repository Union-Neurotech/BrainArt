"""
muse_stream_test.py — minimal BrainFlow Muse connectivity / streaming probe.

Connects to the Muse and, once per second, prints how many samples have
arrived on each preset the board actually supports, plus the latest EEG
sample values. This shows whether the device is streaming at all and, if
so, on which preset. (Muse boards split EEG / accel-gyro / PPG across
separate presets; the original Muse 2016 has no PPG preset.)

Usage:
    python muse_stream_test.py                  # defaults to muse_2016
    python muse_stream_test.py --board muse_2
    python muse_stream_test.py --seconds 20 --timeout 15
    python muse_stream_test.py --serial Muse-XXXX
"""
import argparse, time
import numpy as np
from brainflow.board_shim import (
    BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets, LogLevels
)
from brainflow.exit_codes import BrainFlowError

BOARD_IDS = {
    "muse_2": BoardIds.MUSE_2_BOARD,
    "muse_s": BoardIds.MUSE_S_BOARD,
    "muse_2016": BoardIds.MUSE_2016_BOARD,
}
PRESET_NAME = {
    int(BrainFlowPresets.DEFAULT_PRESET): "DEFAULT(EEG)",
    int(BrainFlowPresets.AUXILIARY_PRESET): "AUX(acc/gyro)",
    int(BrainFlowPresets.ANCILLARY_PRESET): "ANC(PPG)",
}


def supported_presets(bid):
    """Only the presets this board actually defines (avoids 'num_rows not found')."""
    try:
        return [int(p) for p in BoardShim.get_board_presets(bid)]
    except Exception:
        return [int(BrainFlowPresets.DEFAULT_PRESET)]  # safe fallback


def describe(bid, presets):
    print(f"\nBoard id: {int(bid)}   supported presets: "
          f"{[PRESET_NAME.get(p, p) for p in presets]}")
    for p in presets:
        try:
            fs = BoardShim.get_sampling_rate(bid, p)
            eeg = BoardShim.get_eeg_channels(bid, p)
            rows = BoardShim.get_num_rows(bid, p)
            print(f"  {PRESET_NAME.get(p, p):14s} fs={fs:>4} Hz  eeg_rows={eeg}  total_rows={rows}")
        except Exception as e:
            print(f"  {PRESET_NAME.get(p, p):14s} (metadata unavailable: {e})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="muse_2016", choices=list(BOARD_IDS))
    ap.add_argument("--serial", default="")
    ap.add_argument("--mac", default="")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--preset", default="", help="startup Muse preset via other_info, e.g. p20 or p21")
    args = ap.parse_args()

    BoardShim.enable_dev_board_logger()
    BoardShim.set_log_level(LogLevels.LEVEL_INFO)

    bid = BOARD_IDS[args.board]
    params = BrainFlowInputParams()
    params.timeout = args.timeout
    if args.serial:
        params.serial_number = args.serial
    if args.mac:
        params.mac_address = args.mac
    if args.preset:
        params.other_info = args.preset   # docs: other_info carries the startup Muse preset
        print(f"Using startup preset: {args.preset}")

    presets = supported_presets(bid)
    default_preset = int(BrainFlowPresets.DEFAULT_PRESET)
    describe(bid, presets)

    board = BoardShim(bid, params)
    print(f"Preparing session ({args.board}, timeout={args.timeout}s) ...")
    board.prepare_session()
    print("Session ready. Starting stream ...")
    try:
        board.start_stream()   # same call the server uses (default ring buffer)
    except BrainFlowError as e:
        print(f"\n!! start_stream failed: {e}")
        print("   The BLE link is up but the device refused to start streaming.")
        print("   Most common causes on Muse 2016: poor sensor contact, low battery,")
        print("   or a board-id mismatch — try --board muse_2 or --board muse_s.")
        board.release_session()
        return

    eeg_rows = BoardShim.get_eeg_channels(bid, default_preset)
    print(f"Streaming for {args.seconds}s. EEG rows = {eeg_rows}\n")

    try:
        for sec in range(args.seconds):
            time.sleep(1.0)
            parts = [f"[t+{sec+1:>2}s]"]
            for p in presets:
                try:
                    parts.append(f"{PRESET_NAME.get(p, p)}={board.get_board_data_count(p):>5}")
                except Exception:
                    pass
            preview = ""
            try:
                if board.get_board_data_count(default_preset) > 0:
                    d = board.get_current_board_data(1, default_preset)
                    if d.shape[1]:
                        preview = "  EEG µV: " + " ".join(f"{v:8.1f}" for v in d[eeg_rows, -1])
            except Exception:
                pass
            print("  ".join(parts) + preview)
    finally:
        print("\nStopping ...")
        try:
            d = board.get_board_data(preset=default_preset)
            print(f"Total EEG samples captured: {d.shape[1]}")
            if d.shape[1]:
                seg = d[eeg_rows, :]
                print(f"  per-channel mean µV: {np.round(seg.mean(1), 1)}")
                print(f"  per-channel std  µV: {np.round(seg.std(1), 1)}")
                print("  -> non-zero std means real EEG is flowing.")
        except Exception as e:
            print("  (could not read final buffer:", e, ")")
        # the link is often already dead by now ("failed to send command h"),
        # so tear down defensively instead of crashing
        for fn in (board.stop_stream, board.release_session):
            try:
                fn()
            except Exception as e:
                print(f"  ({fn.__name__} skipped: {e})")
        print("Done.")


if __name__ == "__main__":
    main()
