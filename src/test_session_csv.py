"""Session-recording check. Run: python src/test_session_csv.py

Uses the synthetic board's channel layout, so no hardware and no session needed
-- BoardShim's channel getters are static.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardIds, BoardShim

from server import Backend

BOARD = int(BoardIds.SYNTHETIC_BOARD)


def main():
    be = Backend(image_dir=tempfile.mkdtemp(prefix="brainart_test_"))
    be.board_id = BOARD
    be._session_stamp = "20260808_120000"

    rows = BoardShim.get_num_rows(BOARD)
    samples = 100
    data = np.arange(rows * samples, dtype=float).reshape(rows, samples)

    path = be._write_csv(data)
    assert os.path.basename(path) == "brainart_eeg_20260808_120000.csv", path

    df = pd.read_csv(path)
    # Transposed: one row per sample, one column per board channel.
    assert df.shape == (samples, rows), f"expected {(samples, rows)}, got {df.shape}"

    # Named channels survive, and land on the right rows of the matrix.
    for i, nm in zip(BoardShim.get_eeg_channels(BOARD), BoardShim.get_eeg_names(BOARD)):
        assert nm in df.columns, f"missing EEG column {nm}"
        assert np.allclose(df[nm].to_numpy(), data[i]), f"{nm} holds the wrong row"
    assert "timestamp" in df.columns, df.columns.tolist()

    # A second write in the same second must not clobber the first.
    again = be._write_csv(data)
    assert again != path and os.path.exists(path), (path, again)

    print(f"ok: {df.shape[0]} samples x {df.shape[1]} channels -> {os.path.basename(path)}")
    print(f"    columns: {', '.join(df.columns[:8])} ...")


if __name__ == "__main__":
    main()
