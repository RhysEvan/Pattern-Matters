import os
import re

# Folder containing the log files
log_folder = os.path.join("output", "logs", "ViT")
output_folder = os.path.join("output", "loss")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Regular expression pattern to find "loss = [float number]"
loss_pattern = r'loss = ([\d\.]+)'

# Loop through each file in the folder
for log_filename in os.listdir(log_folder):
    if log_filename.endswith('.log'):  # Only process .log files
        log_filepath = os.path.join(log_folder, log_filename)
        
        # Open and read the log file
        with open(log_filepath, 'r') as log_file:
            lines = log_file.readlines()

        # Open an output file to save the extracted losses
        out_filename = '_'.join(log_filename.split('_')[1: 5])
        output_filepath = os.path.join(output_folder, f"{out_filename}_rmse.txt")
        with open(output_filepath, 'w') as output_file:
            for line in lines:
                match = re.search(loss_pattern, line)
                if match:
                    # Write the extracted loss to the output file
                    output_file.write(match.group(1) + '\n')

print("Loss extraction complete!")