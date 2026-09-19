import numpy as np

y = np.load("features/embeddings/binary_labels.npy")
print("Bird:", np.sum(y==1))
print("Noise:", np.sum(y==0))