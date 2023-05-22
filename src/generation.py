# Union Neurotech 2023
# ------------------------------
# Authors:
#   - Leonardo Ferrisi (@leonardoferrisi)
# ------------------------------

# GENERATION METHODS
# ------------------------------
# Description:
#   This file contains the methods used to generate images and reports created from the data collected from 
#   the Electrophysiological Recording Equipment.

# ------------------------------

# Imports
import numpy as np
import os
# ----------------------------------------------------------

# File Utilities
def make_dir(dir_name):
    # Current filepath
    current_filepath = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_filepath)
    output_directory = os.path.join(current_directory, ".." , dir_name)

    if not os.path.exists(output_directory): os.makedirs(output_directory)
    return output_directory

# ----------------------------------------------------------
# Activation Functions (Affect the resulting distribution of the image)

# Note: All activation functions here are dynamically added to numpy_image_nn during construction
#   To add a new activation function write a function here that takes a numpy array 
#   and returns the new activated array. 

def tanh(matrix):
    return np.tanh(matrix)

def sigmoid(matrix):
    return 1.0 / (1 + np.exp(-matrix))

def relu(matrix):
    return np.maximum(matrix, 0, matrix)

def softmax(matrix):
    expo = np.exp(matrix)
    expo_sum = np.sum(np.exp(matrix))
    return expo/expo_sum

def sech(matrix):
    return 1.0 / np.cosh(matrix)

# Feel free to make more !

ACTIVATION_FUNCTIONS = {
                        "tanh"    : tanh, 
                        "sigmoid" : sigmoid, 
                        "relu"    : relu, 
                        "softmax" : softmax, 
                        "sech"    : sech
                        }

# -------------------------------------------------------

# Art Net

class NumpyArtGenerator:
    """ Generates imagery using randomness and a fully connected forward propagation network """

    def __init__(self, resolution, feature_vector, num_layers, layer_width, activation_name, randomness=0.0):
        """Initialize the network.

            resolution          -- tuple resolution of the output network 
            seed                -- seed value used by numpy random 
            num_layers          -- the number of hidden layers in the neural network
            layer_width         -- the number of perceptrons in each hidden layer 
            activation_name     -- name of the activation function used in each hidden layer
        """
        # Flags
        self.complete = False
        self.randomness = randomness
        # ========================================

        self.resolution = resolution


        # Apply Uniqueness from data
        self.seed = self.generate_seed(feature_vector)

        if randomness == 1.0: 
            rng = np.random.default_rng(seed=self.seed) # Doesnt seem to produce reproducable results

            self.rng = rng

        else: 
            # Convert larger seed to being between 0 and 2^32 - 1
            if self.seed > 2**32 -1:
                self.seed = self.seed % (2**32 - 1)
            np.random.seed(self.seed)


        # Define the layers that are being used
        self.num_layers = num_layers
        self.layer_width = layer_width
        
        self.modify_from_metrics(feature_vector)

        # Output Layer of the following sizes correspond to the following values
        # 2 = Black / White
        # 3 = RGB
        # 4 = CMYK
        self.output_layer = 3

        self.__set_activation(activation_name)

    def generate_seed(self, feature_vector):
        # Take all indicies of a feature vector and merge them into a single long integer
        # Example: [0.1, 0.2, 0.3, 0.5] = 01020305
        # This is used to generate a unique seed for each image from the feature vector
        
        # This is done by making a string of all the values in the feature vector
        # Then converting that string to an integer

        seed = ""

        for value in feature_vector:
            value = value * 100 # Scale the value to be between 0 and 100
            # Convert the value to a string, use only the elements before the decimal point
            value = str(value).split(".")[0]
            seed += value
        return int(seed)
        # TODO: Validate that this works

    def modify_from_metrics(self, feature_vector):
        """
        Used EEG ML metrics to modify output image
        """
        concentration = feature_vector[5]
        mindfulness = feature_vector[6]
        restfulness = feature_vector[7]

        # Initialize modifiers
        width_modifier = 0
        layer_modifier = 0
        
        if concentration > restfulness:
            print("Modifying image based on concentration")
            if concentration > 0.6:
                layer_modifier = int(1.0 + (concentration - 0.7)*10)
            
        else:
            print("Modifying image based on restfulness")

            if restfulness >= 0.6:
                layer_modifier = int((mindfulness - 0.7)*-10)    
        
        if mindfulness > 0.6:
            print("Modifying image based on mindfulness")
            width_modifier = int((mindfulness - 0.6)*10)*2
            layer_modifier = int((mindfulness - 0.6)*10)*-1
        
        self.layer_width += width_modifier
        self.num_layers += layer_modifier
        
    def __str__(self):
        """String representation of the network. """

        activation_string = self.activation[0]
        return "-".join([str(self.seed), str(activation_string), str(self.num_layers), str(self.layer_width)])


    def __set_activation(self, activation_name):
        """Set the activation functions for the network. """

        activations_dict = ACTIVATION_FUNCTIONS             # A modification to the original using a predefined dictionary

        if activation_name not in activations_dict:
            raise KeyError("Invalid activation function: " + activation_name + ". Supported activation functions can be found in numpy_activation.py.")

        activation_func = activations_dict[activation_name]
        self.activation = (activation_name, activation_func)

    def __generate_input(self):
        """Generate the x,y coordinate matrices used as input for the network. """
        (ncols, nrows) = self.resolution

        rowmat = (np.tile(np.linspace(0, nrows-1, nrows, dtype=np.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(min(nrows, ncols)/2.0)
        colmat = (np.tile(np.linspace(0, ncols-1, ncols, dtype=np.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(min(nrows, ncols)/2.0)

        inputs = [rowmat, colmat, np.sqrt(np.power(rowmat, 2)+np.power(colmat, 2))]
        return np.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))

    def forward_prop(self, inputs):
        """Run forward propagation on the network"""
        results = inputs

        (ncols, nrows) = self.resolution

        for layer in range(0, self.num_layers):
            print("Current neural layer: " + str(layer+1) + "/" + str(self.num_layers), end='\r')

            if layer == self.num_layers - 1:
                # if self.randomness == 1.0:
                    # W = self.rng.random(size=(results.shape[1], self.output_layer))
                # else:
                W = np.random.randn(results.shape[1], self.output_layer)

            else:
                # if self.randomness == 1.0:
                    # W = self.rng.random(size=(results.shape[1], self.layer_width))
                    # print(W)
                # else:
                W = np.random.randn(results.shape[1], self.layer_width)

            activation_func = self.activation[1]
            results = activation_func(np.matmul(results, W))

        results = (255.0 * results.reshape(nrows, ncols, results.shape[-1])).astype(np.uint8)
        return results


    def print_details(self):
        """Print details of the generator"""

        print("Generator Settings:")
        print(f"    Resolution: {self.resolution}")
        print(f"    Seed: {self.seed}")
        print(f"    Number of Layers: {self.num_layers}")
        print(f"    Hidden Layer Width: {self.layer_width}")
        print(f"    Activation Function: {self.activation[0]}")


    def run(self, verbose):
        """ Run the generator. This includes generating inputs and running the forward propagation network"""

        if verbose:
            self.print_details()

        inputs = self.__generate_input()

        self.results = self.forward_prop(inputs)
        self.complete = True
        return self.results

    def save_image(self, filename, output_directory):
        """
        Generates an image using the data and saves it to the output directory
        """
        import cv2

        if not self.complete:
            raise Exception("Cannot save image before running the generator. Run the generator first using the run() method.")
        
        output_directory_fullpath = make_dir(output_directory)

        print("\nSaving image to " + output_directory + ": " + filename)

        file_full_path = os.path.join(output_directory_fullpath, filename)
        data = self.results
        cv2.imwrite(file_full_path, data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return file_full_path

if __name__ == "__main__":

    # Tests
    # ------------------------------
    # Description:
    #   This file contains the tests for the generation methods
    # ------------------------------

    alpha = 0.3
    beta = 0.2
    gamma = 0.1
    delta = 0.05
    theta = 0.05
    concentration = 0.6
    mindfulness = 0.9
    restfulness = 0.4

    feature_vector = np.array((alpha, beta, gamma, delta, theta, concentration, mindfulness, restfulness))
    n = NumpyArtGenerator(resolution=(512, 512), feature_vector=feature_vector, num_layers=10, layer_width=18, activation_name="tanh")
    n.run(verbose=True)
    n.save_image("test.jpg", "generated//images")