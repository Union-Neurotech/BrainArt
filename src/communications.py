# Union Neurotech 2023
# ------------------------------
# Authors:
#   - Leonardo Ferrisi (@leonardoferrisi)
# ------------------------------

# COMMUNICATIONS METHODS
# ------------------------------
# Description:
# This file contains the methods used to communicate with the Electrophysiological Recording Equipment.


# NOTICE:
# As of current this is being unused. May be integrated in future iteraations

import brainflow

from brainflow import BoardIds, BrainFlowInputParams, BoardShim, BrainFlowError, BrainFlowClassifiers, BrainFlowMetrics

import os
import time

class Comms:

    def __init__(self, board_id, port=None, connect_on_init=True, debug=False):
        
        self.debug = debug

        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        if port != None: params.serial_port = port
    
        self.board = BoardShim(int(board_id), params)

        self.is_connected = False
        if connect_on_init:
            self.connect()
    
    def connect(self):
        """
        Connect to the board
        """
        try:
            self.board.prepare_session()
            self.is_connected = True
        except Exception as e:
            if self.debug:
                print(e)
                print("Error preparing session.")
            return e
    
    def disconnect(self):
        if self.is_connected:
            # Clear the flag even if release_session() raises. Otherwise a failed
            # release leaves this object permanently claiming to be connected,
            # and nothing can recover without restarting the backend.
            try:
                self.board.release_session()
            finally:
                self.is_connected = False

    def start_stream(self, num_samples=450000):
        # Explicit buffer size (BrainFlow's own default) so the caller that saves
        # the session can state the ceiling it is working against.
        self.board.start_stream(num_samples)

    def stop_stream(self):
        self.board.stop_stream()
    
    def get_board_obj(self):
        return self.board
    
    def get_data(self, num_samples=None):
        if num_samples is not None:
            return self.board.get_board_data(num_samples)
        
        print(self.board.get_board_data())
        return self.board.get_board_data()
    
    # save_data() lived here as an empty stub whose `if self.save_data:` tested the
    # method object itself (always truthy). Session recording is real now and lives
    # in Backend._write_csv (server.py), which has the board_id for column names.
