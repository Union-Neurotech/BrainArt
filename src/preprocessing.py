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
from brainflow.ml_model import BrainFlowClassifiers, BrainFlowMetrics, BrainFlowModelParams, MLModel

import numpy as np
import pandas as pd
import os



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

def get_simple_feature_vector(data:pd.DataFrame, boardID:int):
    """
    A simpler method for a getting a general global feature vector

    Parameters:
        :data: A pandas dataframe containing the data to be used for prediction
        :boardID: The ID of the board used to collect the data
    
    Returns:
        :simple_feature_vector: A 8-index array containing the features extracted from the data

        The feature vector contains the following information:
        0: Alpha Bandpower
        1: Beta Bandpower
        2: Delta Bandpower
        3: Theta Bandpower
        4: Gamma Bandpower
        5: Concentration Prediction
        6: Mindfulness Prediction
        7: Relaxation Prediction
    """
    eeg_channels = BoardShim.get_eeg_channels(boardID)
    sampling_rate = BoardShim.get_sampling_rate(boardID)

    if data.shape[0] > data.shape[1]:
        raise Exception("Data must be in the shape (num_channels, num_samples). If Data.shape[0] > Data.shape[1], please transpose the data.")

    # Needed for error avoidance
    data = np.ascontiguousarray(data)

    data = data[:len(eeg_channels)] # Shorten to just our eeg channels

    print(f"Shape is: {data.shape}")
    eeg_indicies = list(range(0, len(eeg_channels)))

    print(f"EEG indices: {eeg_indicies}")


    bands_global = DataFilter.get_avg_band_powers(data, eeg_indicies, sampling_rate, apply_filter=True)

    bands_global = bands_global[0]
    print(f"Average bands: {bands_global}")
    print(f"Average Bands type: {type(bands_global)}")

    # Reorder bands since initial order is delta, theta, alpha, beta, gamma

    # We want alpha, beta, delta, theta, gamma --> see below
    bands_global = np.array([bands_global[2], bands_global[3], bands_global[0], bands_global[1], bands_global[4]])

    # Get ML_Predictions
    concentration = get_concentration_value(data=data, boardID=boardID, sampling_rate=sampling_rate)
    mindfulness = get_mindfulness_value(data=data, boardID=boardID, sampling_rate=sampling_rate)
    relaxation = get_relaxation_value(data=data, boardID=boardID, sampling_rate=sampling_rate)

    # Concatenate all features

    # Normalize bands global
    # bands_global = bands_global/np.linalg.norm(bands_global)

    bands_global = bands_global / np.sum(bands_global)
    
    print(f"Bands Global: {bands_global}")
    
    feature_dict = {
        "bands_global": bands_global,
        "alpha": bands_global[0],
        "beta": bands_global[1],
        "delta": bands_global[2],
        "theta": bands_global[3],
        "gamma": bands_global[4],
        "concentration": concentration[0],
        "mindfulness": mindfulness[0],
        "relaxation": relaxation[0]
    }

    print(f"feature_dict: {feature_dict}")

    simple_feature_vector = np.array((bands_global[0], bands_global[1], bands_global[2], bands_global[3], bands_global[4], concentration[0], mindfulness[0], relaxation[0]))
    print(f"Simple feature vector: {simple_feature_vector}")
    # simple_feature_vector = np.concatenate((bands_global, [concentration, mindfulness, relaxation]))

    return simple_feature_vector


# ML Methods

def get_ML_prediction_value(data, boardID, ML_Model, Classifier, sampling_rate, modelpath="", output_name='probabilities', chunk_size=5):
    """
    Gets the prediction value from a brainflow ML model

    Parameters:
        :data: A pandas dataframe containing the data to be used for prediction
        :ML_Model: The ML Model to be used for prediction
        :Classifier: The classifier to be used for prediction
        :sampling_rate: The sampling rate of the data
        :modelpath: (Relevant only if USERDEFINED) The path to the model to be used for prediction
        :output_name: (Relevant only if USERDEFINED) The name of the output to be used for prediction
        :chunk_size: The size of the chunks to be used for prediction

    Returns:
        :ml_prediction_average: The average prediction value across all chunks
    """
    
    if data.shape[0] > data.shape[1]:
        print(f"Data is of the shape {data.shape}")
        raise Exception("Data must be of the shape (num_samples, num_channels). Please transpose the data before passing it in.")
    
    model_params = BrainFlowModelParams(ML_Model,Classifier)
    if ML_Model == BrainFlowMetrics.USER_DEFINED:
        # Get the path of this file
        this_filepath = os.path.abspath(__file__)
        this_directory = os.path.dirname(this_filepath)
        model_filepath = os.path.join(this_directory, 'models', modelpath)
        # print(f"Model Filepath: {model_filepath}")
        model_params.file = model_filepath
        model_params.output_name = "probabilities"

    ml_prediction = MLModel(model_params)
    ml_prediction.prepare()

    # chunk_sample_size = chunk_size * sampling_rate
    
    # Get the average prediction value across all chunks
    ml_prediction_average = []

    # Get the eeg indicies
    eeg_channels = BoardShim.get_eeg_channels(boardID)
    eeg_indicies = list(range(0, len(eeg_channels)))


    feature_vector = DataFilter.get_avg_band_powers(data, eeg_indicies, sampling_rate, apply_filter=True)
    feature_vector = feature_vector[0] # Get only the first part; the second part is standard deviation which we don't care about right now
    prediction = ml_prediction.predict(feature_vector)

    print(f"Prediction: {prediction}")
    ml_prediction_average.append(prediction)
    
    ml_prediction.release()
    
    # Return the average prediction value across all chunks
    return prediction

def get_concentration_value(data:pd.DataFrame, boardID:int, sampling_rate:int, chunk_size=5):
    """
    Gets the concetration value from the data.
    
    Parameters:
        :data: A pandas dataframe containing the data
        :sampling_rate: The sampling rate of the data
        :chunk_size: The size of the chunks to split the data into

    Returns:
        :concentration_value: A float value between 0 and 1 representing the concentration value
    """    
    return get_ML_prediction_value(data=data, boardID=boardID, ML_Model=BrainFlowMetrics.USER_DEFINED, Classifier=BrainFlowClassifiers.ONNX_CLASSIFIER, \
                                   modelpath='forest_concentration.onnx', sampling_rate=sampling_rate, chunk_size=chunk_size)

def get_mindfulness_value(data:pd.DataFrame, boardID:int, sampling_rate:int, chunk_size=5):
    """
    Gets the mindfulness value from the data.

    
    Parameters:
        :data: A pandas dataframe containing the data
        :sampling_rate: The sampling rate of the data
        :chunk_size: The size of the chunks to split the data into

    Returns:
        :mindfullness_value: A float value between 0 and 1 representing the mindfulness value
    """
    
    return get_ML_prediction_value(data=data, boardID=boardID, ML_Model=BrainFlowMetrics.MINDFULNESS, Classifier=BrainFlowClassifiers.DEFAULT_CLASSIFIER, \
                                   sampling_rate=sampling_rate, chunk_size=chunk_size)

def get_relaxation_value(data:pd.DataFrame, boardID:int, sampling_rate:int, chunk_size=5):
    """
    Gets the relaxation value from the data.

    Parameters:
        :data: A pandas dataframe containing the data
        :sampling_rate: The sampling rate of the data
        :chunk_size: The size of the chunks to split the data into
    
    Returns:
        :relaxation_value: A float value between 0 and 1 representing the relaxation value
    """
    
    return get_ML_prediction_value(data=data, boardID=boardID, ML_Model=BrainFlowMetrics.RESTFULNESS, Classifier=BrainFlowClassifiers.DEFAULT_CLASSIFIER, \
                                   sampling_rate=sampling_rate, chunk_size=chunk_size)

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

    if data.shape[0] > data.shape[1]:
        data = np.transpose(data)

    print(f"Starting Data shape is:  {data.shape}")
    if data.shape[0] != num_eeg_channels:
        data = data[:num_eeg_channels]
    
    band_powers_spatial = np.zeros(num_eeg_channels+1)
    print("Bandpowers spatial is of shape: ", band_powers_spatial.shape)

    overall_bandpower = float(0.0)

    # Apply the bandpass filter to each EEG channel
    print(f"Iterating through channels, we are of shape: {data.shape}")
    print(f"One channel thus looks like: {data[:, 1]}, of shape {data[:, 1].shape}")

    # transpose before we iteratre through channels
    for count, channel in enumerate(eeg_channels):
        
        current_channel = data[count]
        
        print(f"Data shape is {data.shape}")
        print(f"Current channel is {count}")
        print(f"Channel shape is {current_channel.shape}")

        # DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.FIFTY.value)


        # optional detrend
        DataFilter.detrend(current_channel, DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(current_channel, nfft, nfft // 2, sampling_rate,
                                    WindowOperations.BLACKMAN_HARRIS.value)

        actual_bandpower_value = DataFilter.get_band_power(psd, lower_cutoff, upper_cuttoff)

        print("Add to spatial band powers")
        band_powers_spatial[channel] = actual_bandpower_value

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
    
    if __name__ == "__main__":
        # Test the read_file method

        this_filepath = os.path.abspath(__file__)
        thisdirectory = os.path.dirname(this_filepath)

        test_filepath = os.path.join(thisdirectory,"..","test.csv")

        saved_data = DataFilter.read_file(test_filepath)

        restored_df = pd.DataFrame(saved_data)

        # Data we use must be of the shape, (num_channels, num_samples)

        print(restored_df.shape)

        simple_feature_vector = get_simple_feature_vector(restored_df, BoardIds.SYNTHETIC_BOARD.value)

        print(simple_feature_vector)




    # Test the get_bandpower method


    # A test to make sure that larger seeds work
    # for i in range(9999999999999, 100000000000000):
    #     try:
    #         np.random.default_rng(i)
    #     except Exception as e:
    #         print(f"Failure: i is {i} and error is {e}")