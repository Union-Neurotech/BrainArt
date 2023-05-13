# Union Neurotech 2023
# ------------------------------
# Authors:
#   - Leonardo Ferrisi (@leonardoferrisi)
# ------------------------------

# PREPROCESSING METHODS
# ------------------------------
# Description:
#   This file contains the methods used to preprocess Electrophysiological Data collected using methods in `communications.py`.

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, NoiseTypes

import numpy as np
import pandas as pd



# Feature Vector Extraction

def get_feature_vector(data:pd.DataFrame, boardID:int):
    """
    Gets a vector containing features extracted from the data

    Returns:
        :feature_vector: An n-index array containing the features extracted from the data
        :normalized_feature_vector: A normalized version of the feature vector
        :feature_names: A list of the names of the features in the feature vector
        :feature_vector_dict: A dictionary mapping feature names to their respective values
        :normalized feature_vector_dict: A dictionary mapping feature names to their respective values
        :feature_vector_omnibus: A dictionary mapping a feature_vector_dict and normalized feature_vector_dict to each channel
    """
    # For finishing

    # We want a feature vector for each channel, and a feature vector for all channels
    # Each feature vector contains the following information:
    # - Alpha Bandpower
    # - Beta Bandpower
    # - Delta Bandpower
    # - Theta Bandpower
    # - Gamma Bandpower
    # - Concentration
    # - Mindfulness
    # - Relaxation

    # Making an 8-index array for each channel, and an 8-index array for all channels

    # The ML models only take in a 5-index bandpower feature vector, so take that into account
    # The seed that will be generature from this feature fector will incorporate all 8 features

    # generation.py will handle how to manipulate the colors, layering, and other aspects of the image using this feature vector




    # Bandpower makes sure that data is of the right shape but just in case we check here
    

    bands_quantity = 5
    eeg_channels = BoardShim.get_eeg_channels(boardID)
    eeg_channel_names = BoardShim.get_eeg_names(boardID)
    num_eeg_channels = len(eeg_channels)
    num_channels, num_samples = data[:len(num_eeg_channels)].shape

    if data.shape[0] != len(num_eeg_channels):
        data = data[:len(num_eeg_channels)]

    global_feature_vector = np.zeros(bands_quantity)
    feature_names = [ "alpha", "beta", "delta", "theta","gamma"] # These are stored in order of prominence in the EEG spectrum

    feature_vector_dict = {}

    feature_omnibus = {}

    

    # Get Bandpower Feature Vector for All channels
    for channel in range(num_channels):

        current_eeg_channel = eeg_channels[channel]

        overall_alpha, alpha_channelpairs = get_alpha_band(data[channel], boardID)
        overall_beta, beta_channelpairs   = get_beta_band(data[channel], boardID)
        overall_delta, delta_channelpairs = get_delta_band(data[channel], boardID)
        overall_theta, theta_channelpairs = get_theta_band(data[channel], boardID)
        overall_gamma, gamma_channelpairs = get_gamma_band(data[channel], boardID)

        # Add to global feature vector
        local_feature_vector = np.array([overall_alpha, overall_beta, overall_delta, overall_theta, overall_gamma])
        global_feature_vector = (global_feature_vector + local_feature_vector)/(channel+1)
    



    # Get Bandpower Feature Vector for each channel, stored in a dictionary pairing them with each name

    # Get Concentration, Mindfullness, and Relaxation feature vector

    # Return the concatenation of all feature vectors
    pass


# ML Methods

def get_concentration_value(data:pd.DataFrame):
    """
    Gets the concetration value from the data
    """
    pass

def get_mindfullness_value(data:pd.DataFrame):
    """
    Gets the mindfulness value from the data
    """
    pass

def get_relaxation_value(data:pd.DataFrame):
    """
    Gets the relaxation value from the data
    """
    pass

# Bandpower Extraction Methods

def get_bandpower(data: pd.DataFrame, lower_cutoff:float, upper_cuttoff:float, boardID:int, useNotch:bool=True):
    """
    Apply a bandpass filter and return the estimated power of the signal in said band

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :lower_cutoff: The lower cutoff frequency of the bandpass filter
        :upper_cutoff: The upper cutoff frequency of the bandpass filter
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise

    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    """
    sampling_rate = BoardShim.get_sampling_rate(boardID)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    # Evaluate that we are only using EEG channels
    eeg_channels = BoardShim.get_eeg_channels(boardID)
    channel_names = BoardShim.get_eeg_names(boardID)
    num_eeg_channels = len(eeg_channels)

    if data.shape[0] != len(num_eeg_channels):
        data = data[:len(num_eeg_channels)]
    
    band_powers_spatial = np.zeros(num_eeg_channels)

    overall_bandpower = float(0.0)

    # Apply the bandpass filter to each EEG channel
    for channel in range(num_eeg_channels):

        # Vital Notch Filter to remove environmental noise
        DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)

        # optional detrend
        DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, sampling_rate,
                                    WindowOperations.BLACKMAN_HARRIS.value)

        actual_bandpower_value = DataFilter.get_band_power(psd, lower_cutoff, upper_cuttoff)

        band_powers_spatial[channel] = psd

        overall_bandpower = (overall_bandpower + actual_bandpower_value)/(channel+1)
    
    band_power_channel_pairs = {}

    for channelName, spatialBandPower in zip(channel_names, band_powers_spatial):
        band_power_channel_pairs[channelName] = spatialBandPower

    return overall_bandpower, band_power_channel_pairs

def get_alpha_band(data, boardID, useNotch=True):
    """
    Get the alpha bandpower of the data

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise
    
    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    
    *See get_bandpower for more information*
    """
    return get_bandpower(data, 7.0, 13.0, boardID, useNotch)

def get_beta_band(data:pd.DataFrame, boardID:int, useNotch:bool=True):
    """
    Get the beta bandpower of the data

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise
    
    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    
    *See get_bandpower for more information*
    """
    return get_bandpower(data, 13.0, 30.0, boardID, useNotch)

def get_delta_band(data:pd.DataFrame, boardID:int, useNotch:bool=True):
    """
    Get the delta bandpower of the data

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise
    
    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    
    *See get_bandpower for more information*
    """
    return get_bandpower(data, 0.5, 4.0, boardID, useNotch)

def get_theta_band(data:pd.DataFrame, boardID:int, useNotch:bool=True):
    """
    Get the theta bandpower of the data

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise
    
    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    
    *See get_bandpower for more information*
    """
    return get_bandpower(data, 4.0, 7.0, boardID, useNotch)

def get_gamma_band(data:pd.DataFrame, boardID:int, useNotch:bool=True):
    """
    Get the gamma bandpower of the data

    Parameters:
        :data: The (raw) data to be filtered and analyzed
        :boardID: The ID of the board used to collect the data, used to determine the sampling rate
        :useNotch: Whether or not to use a notch filter to remove environmental noise
    
    Returns:
        :overall_bandpower: The average bandpower across all channels
        :band_power_channel_pairs: A dictionary mapping channel names to their respective bandpower
    
    *See get_bandpower for more information*
    """
    return get_bandpower(data, 30.0, 100.0, boardID, useNotch)




if __name__ == "__main__":
    
    pass

    # Test the get_bandpower method


    # A test to make sure that larger seeds work
    # for i in range(9999999999999, 100000000000000):
    #     try:
    #         np.random.default_rng(i)
    #     except Exception as e:
    #         print(f"Failure: i is {i} and error is {e}")