# Union Neurotech 2023
# ------------------------------
# Authors:
#   - Leonardo Ferrisi (@leonardoferrisi)
# ------------------------------

# RUN METHODS
# ------------------------------
# Description:
#   This file contains the methods to run the BrainArt program as an application.

from browser_ui import BrowserUI
import os

if __name__ == "__main__":
    os.system('cls')
    webapp = BrowserUI(title="Brain Generated Artwork Prototype")