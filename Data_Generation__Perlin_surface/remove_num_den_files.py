import os

# Specify the directory path
directory = 'data'

# Iterate over all the files in the directory
for filename in os.listdir(directory):
    # Check if "_num_" or "_den_" is in the filename
    if '_nums_' in filename or '_dens_' in filename:
        file_path = os.path.join(directory, filename)  # Full path to the file
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")