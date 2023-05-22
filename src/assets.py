from brainflow import BoardIds
import os
board_id_pairs ={
    "Synthetic"                        : {"id":BoardIds.SYNTHETIC_BOARD.value  , "using_port": False},
    "Muse2"                            : {"id":BoardIds.MUSE_2_BLED_BOARD.value, "using_port": True },
    "Muse2016"                         : {"id":BoardIds.MUSE_2016_BLED_BOARD   , "using_port": True },
    "OpenBCI Cyton (8-channels)"       : {"id":BoardIds.CYTON_BOARD.value      , "using_port": True },
    "OpenBCI Cyton-Daisy (16-channels)": {"id":BoardIds.CYTON_DAISY_BOARD.value, "using_port": True },
    "OpenBCI Ganglion"                 : {"id":BoardIds.GANGLION_BOARD.value   , "using_port": True }
}

def get_user_data():
    """
    Retrieve the contents of user_data.dat
    """
     # get the path of this file
    this_filepath = os.path.abspath(__file__)

    # get the directory of this file
    this_directory = os.path.dirname(this_filepath)
    local_data_path = os.path.join(this_directory, "local_data", "user_data.dat")

    return local_data_path

def read_userdata():
    """
    Reads the contents of local_data/userdata.dat
    """
    data = {}
    
    userdata_path = get_user_data()

    with open(userdata_path, "r") as f:
        for line in f.readlines():
            var, val = line.split(": ")
            data[var] = val.split("\n")[0]
    return data

def update_userdata(variable:str, value):

    userdata_path = get_user_data()

    # assert that variable exists

    data = read_userdata()

    # wipe the file
    
    open(userdata_path, "w").close()

    assert variable.upper() in data.keys()

    # change it 

    data[variable] = str(value)

    with open(userdata_path, "w") as f:
        for key, value in data.items():
            content = key.upper() + ": " + value
            f.writelines(content)

if __name__ == "__main__":
    userdata = read_userdata()

    print(userdata)