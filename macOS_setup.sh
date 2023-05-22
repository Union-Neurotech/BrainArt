#!/bin/bash

    # To use this script, follow these steps:
    # 1. Open a text editor (such as TextEdit or Sublime Text) on your macOS system.
    # 2. Copy and paste the above script into the text editor.
    # 3. Save the file with a ".sh" extension, such as "setup.sh".
    # 4. Open Terminal on your macOS system.
    # 5. Navigate to the directory where you saved the script using the cd command. For example, if you saved it on your desktop, you can use cd ~/Desktop.
    # 6. Make the script executable by running the command: chmod +x setup.sh.
    # 7. Run the script by entering its name: ./setup.sh.
    # 8. The script will execute and display the output in the Terminal, showing the value of the BRAINART environment variable set to the path of the script directory.

# Get the path of the directory containing the setup script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the BRAINART environment variable to the script directory
export BRAINART="$script_dir"

# Display the value of BRAINART
echo "BRAINART environment variable set to: $BRAINART. You can close this now."
