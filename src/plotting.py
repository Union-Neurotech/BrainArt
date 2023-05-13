import matplotlib.pyplot as plt
import numpy as np
import os
import brainflow
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, AggOperations, FilterTypes, NoiseTypes
import pandas as pd
from alive_progress import alive_bar

import matplotlib.pyplot as plt

def generate_raw_plot(filename:str, boardID:int, data:pd.DataFrame, transpose:bool=False, descale_weight:int=10000, title:str='60 seconds of Raw EEG Data', show=True, save=False, show_progress=True):    
    """
    Generate a plot of the raw EEG data

    Data should be formatted in the format data[channels, :] - otherwise, set transpose to True
    """

    if transpose: data = np.transpose(data) # data is typically stored in a trasnposed format

    if type(data) is not pd.DataFrame:
        data = pd.DataFrame(data)

    # print(f"Data shape is {data.shape}")
    # data = data[channels, :]
    num_channels, num_samples = data.shape

    # Calculate the sampling rate (assuming the data is recorded at 250 Hz)
    sampling_rate = BoardShim.get_sampling_rate(boardID)

    # Create a time vector for the x-axis
    time = np.arange(num_samples) / sampling_rate

    # print(f"Time shape is {time.shape}")

    channels = BoardShim.get_eeg_channels(boardID)
    boardINFO = BoardShim.get_board_descr(boardID)
    channelNames = boardINFO["eeg_names"]

    if type(channelNames) is str: channelNames = channelNames.split(",")

    # Get only relevant data
    print(f"Found {len(channels)} channels")

    
    # Filter the data
    for i in range(num_channels):
        channel = data[i]
        # import numpy as np

        channel = np.ascontiguousarray(channel)

        DataFilter.perform_bandpass(channel, sampling_rate, 2.0, 50.0, 4, FilterTypes.BESSEL.value, 0)
        DataFilter.perform_highpass(channel, sampling_rate, 2.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_lowpass(channel, sampling_rate, 50.0, 5, FilterTypes.CHEBYSHEV_TYPE_1.value, 1)
        DataFilter.remove_environmental_noise(channel, sampling_rate, NoiseTypes.FIFTY.value)

        DataFilter.perform_rolling_filter(channel, 3, AggOperations.MEAN.value)

        # update data with filtered channel
        data[i] = channel


    weight = 1/descale_weight
    data = data * weight # scale down eeg

    # set the y-axis limits to accommodate 32 channels
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(0, len(channels)+1)



    relevant_data = data[:len(channels)]
    
    print("Generating Plot. Please hold...")
    
    if show_progress:
        # Generate all channel rows for plot
        with alive_bar(len(relevant_data)) as bar:
            for i, channel in relevant_data.iterrows():
                # print(f"i is {i+1}, Current Channel: {channelNames[i]}")

                ax.plot(time, channel+(i+1), label=f'{channelNames[i]}')
                bar()
    
    else:
        for i, channel in relevant_data.iterrows():
            # print(f"i is {i+1}, Current Channel: {channelNames[i]}")

            ax.plot(time, channel+(i+1), label=f'{channelNames[i]}')

    # Plot the EEG data
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Channels')
    ax.set_title(title)
    ax.legend()

    # print("SAVING PLOT")
    if save: 
        print("Saving Plot...")
        plt.savefig(filename)
    if show: plt.show()
    # plt.clf()

    return fig, ax

if __name__ == "__main__":
    
    # Local Filepath
    filepath = os.path.join(os.path.dirname(__file__), "..", "test.csv")
    print(filepath)

    restored_data = DataFilter.read_file('test.csv')
    restored_df = pd.DataFrame(restored_data)
    
    print(restored_df.shape)

    generate_raw_plot(filename="test.png", boardID=brainflow.BoardIds.SYNTHETIC_BOARD.value, data=restored_df, transpose=False, title="60 seconds of Raw EEG Data", show=True, save=True, show_progress=True)


