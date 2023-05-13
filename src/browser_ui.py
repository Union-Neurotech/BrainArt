import streamlit as st
import brainflow
import os
from assets import board_id_pairs, read_userdata, update_userdata

# local imports
# from communications import Comms

import brainflow
from brainflow import BoardIds, BrainFlowInputParams, BoardShim, BrainFlowError, BrainFlowClassifiers, BrainFlowMetrics, DataFilter

from plotting import generate_raw_plot

import time

import pandas as pd

# Prototype methods for a web-browser interface for Brain Generated Artwork
# - Leonardo Ferrisi

class BrowserUI:

    def __init__(self, title="Brain Generated Artwork Prototype", debug=False):
        self.debug = debug
        self.userdata = read_userdata()

        self.current_working_directory = os.getcwd()
        self.current_file_path = os.path.abspath(__file__)
        self.current_file_path_directory = os.path.dirname(self.current_file_path)


        from PIL import Image
        logo_path = os.path.join(self.current_file_path_directory, "local_assets", "logo.png")

        LOGO = Image.open(logo_path)
        st.image(image=LOGO, caption="Union Neurotech 2023", width=200)
        st.title(title)

        if 'connected' not in st.session_state:
            st.session_state['connected'] = False

        self.connected = st.session_state['connected']
        
        if 'streaming' not in st.session_state:
            st.session_state['streaming'] = False
        self.streaming = st.session_state['streaming']

        if 'recording_data' not in st.session_state:
            st.session_state['recording_data'] = None

        if 'EEG_DATA' not in st.session_state:
            st.session_state['EEG_DATA'] = None
        self.EEG_DATA = st.session_state['EEG_DATA']

        if 'EEG_CHANNELS' not in st.session_state:
            st.session_state['EEG_CHANNELS'] = None
        self.EEG_CHANNELS = st.session_state['EEG_CHANNELS']

        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        self.show_prompts()

    
    def connect_board(self, board_id, port=None):
        """
        Connect a board and initiate it as a BoardShim Object
        """
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        if port != None: params.serial_port = port
        
        if ('BOARD' not in st.session_state) or (st.session_state['BOARD'] == None):
            st.session_state['BOARD'] = BoardShim(int(board_id), params)

        self.BOARD = st.session_state['BOARD']
        self.BOARD.prepare_session()
        
        st.session_state['EEG_CHANNELS'] = self.BOARD.get_eeg_channels(int(board_id))
        self.EEG_CHANNELS = st.session_state['EEG_CHANNELS']

        try:
            with st.expander("[For Debugging] Session State:"):
                st.write(st.session_state)
            with st.expander("[For Debugging] Board Description:"):
                board_descr = BoardShim.get_board_descr(board_id)
                st.write("Board Description: ", board_descr)
        except:
            st.text("No session state available. or failed to print :(")
    def disconnect_board(self):
        """
        Disconnect the board and set the user data to None
        """
        if self.connected:
            self.BOARD = st.session_state['BOARD']
            self.BOARD.release_all_sessions()
            st.session_state['BOARD'] = None
            self.connected = False

    def change_connection_status(self):
        """
        Change the connection status of the board
        """
        if self.connected:
            self.connected = False
        else: 
            self.connecred = True

    def stream_data(self):
        """
        Stream data from an initialized BoardShim Object
        """
        self.BOARD = st.session_state['BOARD']
        self.BOARD.start_stream()

        # Add a progress bar
        progress_bar = st.progress(0)
        durations = self.collection_time
        iterations = 100
        sleep_time = durations / iterations
        start_time = time.time()
        for i in range(iterations):
            time_elapsed = time.time() - start_time
            label = f"{time_elapsed:.2f} seconds elapsed"
            progress_bar.progress(i + 1, text=label)
            time.sleep(sleep_time)

        user_data = self.BOARD.get_board_data()

        # Overwrite EEG_DATA
        st.session_state['EEG_DATA'] = user_data

        self.BOARD.stop_stream()

        st.success("Task completed! Data collected.")
        st.info("If you would like to record again, please click the **R**-key to re-run before clicking `START` again.")
        return user_data
    
    def generate_feature_vector(self, data):
        """
        Generate a vector of features of data
        """
        from preprocessing import get_simple_feature_vector
        
        feature_vector = get_simple_feature_vector(data=data, boardID=self.CURRENT_BOARD_ID)

        return feature_vector

    def generate_image(self, features):
        """
        Generates an image from features of EEG
        """
        pass

    def show_prompts(self):
        """
        Show all prompts for starting the Brain Artwork program
        """

        # Show buttons for using either prerecorded data or live data
        data_source = st.radio("Data Source", ("Live", "Prerecorded"), horizontal=True, help="Select the source of EEG data")

        # If prerecorded data is selected, show a file upload button
        if data_source == "Prerecorded":
            st.info("Currently this code only supports LIVE. Prerecorded data usage will be added soon. :smile:")
            uploaded_file = st.file_uploader("Choose a file", type="csv")
            if uploaded_file is not None:
                st.write("file uploaded.")

        if data_source == "Live":
            st.write("Live data selected.")
            
            # Continue with Brain Art - Generation. Backend MUST be running.

            # --------------------------------------------------------------
            # Board selection:

            board_select = st.selectbox(
                'Which board are you using?',
                ('Synthetic', 'Muse2', 'Muse2016', 'OpenBCI Cyton (8-channels)', 'OpenBCI (16-channels)', 'OpenBCI Ganglion', ), 
                help="Select the board you are using to record EEG data.")

            st.write('You selected:', board_select)

            if self.debug:
                st.write("Board ID:", board_id_pairs[board_select]["id"])
                st.write("Using port?:", board_id_pairs[board_select]["using_port"])

            self.CURRENT_BOARD = board_select

            # --------------------------------------------------------------
            # Port Selection:
            self.requires_port = board_id_pairs[board_select]["using_port"]
            if self.requires_port:
                st.write("The current board requires a port to be specified.")
                current_port = self.userdata["PORT"]

                if self.debug: st.write("Port:", current_port)

                port_select = st.text_input("The currently saved port is: ", current_port, help="Enter the USB Serial port your board is connected to.")

                if port_select != current_port:
                    try: 
                        update_userdata("PORT", port_select)
                    except:
                        st.write("Error updating port in userdata.dat. Using locally defined PORT")

                self.PORT = port_select

            # --------------------------------------------------------------
            # Configure Image Output Directory
            default_image_directory = os.path.join(self.current_working_directory, "generated", "images")

            self.image_directory = st.text_input("Current Image Output Directory: ", default_image_directory, help="Enter the directory where you would like to save generated images. Change nothing if you are fine with the default configuration.")

            # --------------------------------------------------------------
            # User Data Prompt
            st.divider()
            st.write("#### User Data Consent")
            default_csv_directory = os.path.join(self.current_working_directory, "data")
            # default_contact = self.userdata["CONTACT"]
            default_contact = "unionneurotech@gmail.com"
            st.info(f"Union Neurotech conducts ongoing research in building better Brain Computer Interfaces. \
                         By consenting to participate in this research, you agree to allow Union Neurotech to use your data \
                         for research purposes. \nYour data will be anonymized and will not be shared with any third parties. \
                         \n With your data we can train machines to get better at understanding and interpreting the human mind.\
                         If you have any questions, please contact us at {default_contact}.\
                         \nThis has no affiliation with the Union College Undergraduate Research Program as of current and functions as independent research.")
            collect_data_choice = st.radio("Do you consent to participate in this research?", ("No", "Yes"), help="Select Yes to consent to participate in this research.", horizontal=True)

            if collect_data_choice == "Yes":
                self.collect_data = True
                st.write("Thank you for choosing to participate in our research!")
                st.text_input("Please enter your name: ", help="Enter your name to be used in the data collection process.")
                st.text_input("Please enter your email: ", help="Enter your email to be used in the data collection process.")
                st.write("These will be used to keep track of who has provided data and keep track of affirmative consent. Your contact information\
                         will not be in any way connected to your data. This is only for our records. There will be no way of linking your data back to you.")
            else:
                st.info("By not consenting, You will still be able to generate images, but your data will not be saved.")
            # --------------------------------------------------------------
            # Connect Board
            st.divider()
            port_param = None
            if self.requires_port: port_param = self.PORT
            self.CURRENT_BOARD_ID = board_id_pairs[self.CURRENT_BOARD]["id"]
            

            if self.debug: st.write("##### :green[Successfully created board object.]")

    
            if not self.connected:
                st.write("### Connect to the Streaming Device.")
                if st.button(label="Connect", help="Connect to the Streaming Device. This may take a few seconds. You may want to rerun with `R` after clicking."):
                # Connect to Board
                    try:
                        self.connect_board(self.CURRENT_BOARD_ID, port_param)
                        st.session_state['connected'] = True
                        self.connected = True
                        st.success(f"Successfully connected `{self.CURRENT_BOARD}` Streaming Device.]")

                        st.info(f"Currently has {len(self.EEG_CHANNELS)} channels.")
                    except Exception as e:
                        st.write("##### :red[Failed to connect to Streaming Device. Please check your connection and try again.]")
                        st.error(e)
            else:
                st.write("### Disconnect from the Streaming Device.")
                if st.button(label="Disconnect", help="Disconnect from the Streaming Device. You may want to rerun with `R` after clicking"):
                    self.disconnect_board()
                    st.session_state['connected'] = False
                    try:
                        st.info(f"Successfully disconnected `{self.CURRENT_BOARD}` Streaming Device.]")
                    except:
                        st.write("##### :red[Failed to disconnect from Streaming Device. Please check your connection and try again.]")
                    self.connected = False
                if st.session_state['connected']: st.success(f"Successfully connected `{self.CURRENT_BOARD}` Streaming Device.]")
                try:
                    with st.expander("[For Debugging] Session State:"):
                        st.write(st.session_state)
                    with st.expander("[For Debugging] Board Description:"):
                        board_descr = BoardShim.get_board_descr(self.CURRENT_BOARD_ID)
                        st.write("Board Description: ", board_descr)
                except:
                    st.text("No session state available. or failed to print :(")


            # --------------------------------------------------------------
            # Start Data Collection
            if self.connected:
                st.divider()
                st.write("### Begin Data Collection.")
                st.info("""
                Pro Tip! \n
                If you dont want to disconnect the device, you can just re-run and start the data collection again. Click 'R' to re-run.""")
                self.collection_time = st.number_input("Enter the number of seconds you want to collect data for: ", min_value=1, max_value=600, value=60, step=1, help="Enter the number of seconds you want to collect data for.")
                if st.button(label="Start", help="Start collecting data from the Streaming Device."):
                    

                    data = self.stream_data()
                    
                    st.divider()
                    st.markdown("### Raw Data: ")
                    st.info("The following is the raw data output from the board. This is the data that is processed \
                        further used to generate the artwork.")

                    try:
                        with st.expander("Checkout raw data."):
                            st.write("#### Raw Data")
                            st.info(f"The y axis is the channels in all data. The x axis is the time samples. Note that we are only intersted in the first {len(self.EEG_CHANNELS)} channels as those correspond to EEG. The rest are Accelorometer Data, Battery Level, and more!")
                            st.dataframe(data)

                    except Exception as e:
                        st.error(e)

                    # Generate Plot:

                    descale_weight = 10000

                    if self.CURRENT_BOARD.lower() == "synthetic":
                        descale_weight = 1000
                    
                    fig, ax = generate_raw_plot(boardID=self.CURRENT_BOARD_ID, data=data, transpose=False, title="Raw EEG Data Plot", show=False, save=False, descale_weight=descale_weight, filename="raw_plot.png", show_progress=False)

                    # Display the plot in Streamlit
                    
                    st.pyplot(fig)
                    st.info("This is what the data looks like before we filter it and turn it into an image!")
                    
                    st.divider()

                    # Get the features from data

                    feature_vector = self.generate_feature_vector(data)

                    # Display the feature vector
                    st.write("#### Feature Vector")
                    st.info("This is the feature vector that is used to generate the artwork. It is a 1D array of numbers that represent the data collected from the board.")

                    gamma_band = feature_vector[4]

                    progress_bar = st.progress(0, text="Gamma Band")
                    for i in range(int(gamma_band * 100)):
                        progress_bar.progress(i, text=f"Gamma Band: {i}%")

                    beta_band = feature_vector[1]

                    progress_bar = st.progress(0, text="Beta Band")
                    for i in range(int(beta_band * 100)):
                        progress_bar.progress(i, text=f"Beta Band: {i}%")

                    alpha_band = feature_vector[0]

                    progress_bar = st.progress(0, text="Alpha Band")
                    for i in range(int(alpha_band * 100)):
                        progress_bar.progress(i, text=f"Alpha Band: {i}%")

                    theta_band = feature_vector[2]
                    
                    progress_bar = st.progress(0, text="Theta Band")
                    for i in range(int(theta_band * 100)):
                        progress_bar.progress(i, text=f"Theta Band: {i}%")

                    delta_band = feature_vector[3]

                    progress_bar = st.progress(0, text="Delta Band")
                    for i in range(int(delta_band * 100)):
                        progress_bar.progress(i, text=f"Delta Band: {i}%")

                    concentration_prediction = feature_vector[5]

                    progress_bar = st.progress(0, text="Concentration Prediction: 0%")
                    for i in range(int(concentration_prediction * 100)):
                        progress_bar.progress(i, text=f"Concentration Prediction: {i}%")

                    mindfulness_prediction = feature_vector[6]

                    progress_bar = st.progress(0, text="Mindfulness Prediction: 0%")
                    for i in range(int(mindfulness_prediction * 100)):
                        progress_bar.progress(i, text=f"Mindfulness Prediction: {i}%")

                    relaxation_prediction = feature_vector[7]

                    progress_bar = st.progress(0, text="Relaxation Prediction: 0%")
                    for i in range(int(relaxation_prediction * 100)):
                        progress_bar.progress(i, text=f"Relaxation Prediction: {i}%")

                    # Handle saving data
                    # TODO: Handle Saving data


                    image_path = self.generate_image(feature_vector)

                    # Handle displaying image

                    # self.display_artwork(image_path)

    def display_artwork(self, artwork):
        pass
        # st.image(artwork)

if __name__ == "__main__":

    webapp = BrowserUI(title="Brain Generated Artwork Prototype")