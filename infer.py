# infer.py
# Written by Dr. Matthew Smith, Swinburne University of Technology
# Load a precomputed keras model and its weights for a single inference
# USAGE: python infer.py ID where ID is the ID of the training file
# we wish to load.

# Import modules
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from utilities import *
import sys

# Parse the input
no_arg = len(sys.argv)
if (no_arg == 2):
        ID = int(sys.argv[1])
else:
     	print("Usage: python view.py <Data_ID>")
        print("where Data_ID is a number.")
        print("Example: python infer.py 2")
        print("  -- will load X_2.dat and infer its class")
        ID = 2

print("Loading file = " + str(ID))

# We still need to know how long the time series is
N_sequence = 128     # Length of each piece of data

# Create variables for use while inferencing.
# Keeping it in array form; you might want to inference
# multiple data sets later.
X_infer = np.empty([1,N_sequence])
Y_infer = np.empty(1)

# We can take this data from anywhere - let's load one of the training sets
X_infer[0,], Y_infer[0]  = read_test_data(ID, N_sequence)

# Load the JSON file
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# Set up the neural layer configuration in the model
model = model_from_json(loaded_model_json)
# Load the weights into the model
model.load_weights("model.h5")
# Compile it
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Now try classifying the single data file we loaded
Class_infer = model.predict_classes(X_infer)

# Compute the class predictions - shouldn't be used as certainties.
Class_prob = model.predict(X_infer)

print("The predicted class is %d" % Class_infer[0])
print("Class Predictions: Class 0 = %f, Class 1 = %f" % ((1.0-Class_prob[0]), Class_prob[0]))
print("The actual loaded class is %d" % Y_infer[0])

