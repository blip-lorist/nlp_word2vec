import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split

# Read JSON data to data frames
data_frame_clothing = pd.read_json("reviews_Clothing_Shoes_and_Jewelry_5.json", lines = True)
data_frame_sports = pd.read_json("reviews_Sports_and_Outdoors_5.json", lines = True)

data_frame = pd.concat([data_frame_clothing, data_frame_sports])

# Retreive text from the data frame
review_text = data_frame["reviewText"].values

