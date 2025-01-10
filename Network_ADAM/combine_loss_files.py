import os
import glob

# Folder containing the files
folder_path = "output/loss"
network = "UNet"

# Get all the text files in the folder
file_list = glob.glob(os.path.join(folder_path, f"lines_*p_*_{network}_rmse.txt"))

# Create a dictionary to hold content for each k value
content_dict = {}

# Read and group the contents by k value
for file_path in file_list:
    file_name = os.path.basename(file_path)
    
    # Extract k and n values
    k_value = file_name.split('_')[1][:-1]  # Extract the k value
    n_value = file_name.split('_')[2]       # Extract the n value
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Append the content to the respective k value in the dictionary
    if k_value not in content_dict:
        content_dict[k_value] = []
    
    # Add the lines to the list as columns
    if len(content_dict[k_value]) == 0:
        content_dict[k_value] = [line.strip() for line in lines]
    else:
        for i, line in enumerate(lines):
            content_dict[k_value][i] += "\t" + line.strip()

    # Remove the original source file
    os.remove(file_path)

# Write the combined content to new files
for k_value, combined_lines in content_dict.items():
    output_file = os.path.join(folder_path, f"lines_{k_value}p_{network}_rmse.txt")
    
    with open(output_file, 'w') as file:
        for line in combined_lines:
            file.write(line + "\n")

print("Files combined successfully!")