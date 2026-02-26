import birdnet
import os

model = birdnet.load("acoustic", "2.4", "tf")

test_file = "data/IBC53/Corvus_splendens/example.wav"  # replace with real path

predictions = model.predict(test_file)
print(predictions.head())
