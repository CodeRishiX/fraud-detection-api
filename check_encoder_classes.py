import numpy as np

# Load the encoder classes
try:
    classes = np.load("encoder_classes.npy", allow_pickle=True)
    print("Valid transaction types (encoder classes):")
    print(classes)
except Exception as e:
    print(f"Error loading encoder_classes.npy: {e}")