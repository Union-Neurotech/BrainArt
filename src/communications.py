# Union Neurotech 2023
# ------------------------------
# Authors:
#   - Leonardo Ferrisi (@leonardoferrisi)
# ------------------------------

# COMMUNICATIONS METHODS
# ------------------------------
# Description:
# This file contains the methods used to communicate with the Electrophysiological Recording Equipment.

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
            self.board.release_session()
            self.is_connected = False

    def start_stream(self):
        self.board.start_stream()

    def stop_stream(self):
        self.board.stop_stream()
    
    def get_board_obj(self):
        return self.board
    
    def get_data(self, num_samples=None):
        if num_samples is not None:
            return self.board.get_board_data(num_samples)
        
        print(self.board.get_board_data())
        return self.board.get_board_data()
    
    def save_data(self):
        if self.save_data:
            # save data as csv and edf files
            # self.board.save_board_data_as_csv("board_data.csv", "w")
            pass
