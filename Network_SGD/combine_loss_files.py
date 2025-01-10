import os
import glob

# Folder containing the files
folder_path = os.path.join("output", "loss")
network = "UNet"

# Get all the text files in the folder
file_list = glob.glob(os.path.join(folder_path, f"{network}_lines_*p_*_rmse.txt"))

# Create a dictionary to hold content for each k value
content_dict = {}

# Read and group the contents by k value
for file_path in file_list:
    file_name = os.path.basename(file_path)
    
    # Extract k and n values
    p_value = int(file_name.split('_')[2][:-1])  # Extract the k value
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Append the content to the respective k value in the dictionary
    if p_value not in content_dict:
        content_dict[p_value] = []
    
    # Add the lines to the list as columns
    if len(content_dict[p_value]) == 0:
        content_dict[p_value] = [line.strip() for line in lines]
    else:
        for i, line in enumerate(lines):
            content_dict[p_value][i] += "\t" + line.strip()


# Create directory for the combined files
combined_path = os.path.join(folder_path, network)
os.makedirs(combined_path, exist_ok=True)     

# Write the combined content to new files
for p_value, combined_lines in content_dict.items():
    output_file = os.path.join(folder_path, network, f"lines_{p_value}p_rmse.txt")
    
    with open(output_file, 'x') as file:
        for line in combined_lines:
            file.write(line + "\n")

print("Files combined successfully!")

answer = input("Remove source files? (y/n): ")
if answer.lower() == "y":
    for file_path in file_list:
        # Remove the original source file
        os.remove(file_path)
    print("Files removed!")
