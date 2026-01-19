import h5py

# Path to your Brain Tumor MAT file
path = r"C:\Users\Tanishqa More\OneDrive\Documents\quantum_image_representation\datasets\brain_tumor\2299.mat"

# Open the MAT file
with h5py.File(path, 'r') as f:
    # Function to print all keys (nested as well)
    def print_keys(name, obj):
        print(name, type(obj))
    f.visititems(print_keys)
