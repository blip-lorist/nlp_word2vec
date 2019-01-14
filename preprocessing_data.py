import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split

# Snagged this regex cleaning script from @nateraw
import data_helper

from build_word2vec import build_word2vec


"""
Output:
    pivot_words: each one repeats four times since the window size is 2
    Example: [2, 2, 2, 2]

    target_words: negative and positive window target words for skip-gram
    Example: [0.0, 1.0, 3.0, 4.0] should not include the pivot of 2
"""

# Read JSON data to data frames
data_frame_clothing = pd.read_json("reviews_Clothing_Shoes_and_Jewelry_5.json", lines = True)
data_frame_sports = pd.read_json("reviews_Sports_and_Outdoors_5.json", lines = True)

data_frame = pd.concat([data_frame_clothing, data_frame_sports])

# Reduce size temporarily while iterating
data_frame = data_frame[:1000]

# Retreive text from the data frame
review_text = data_frame["reviewText"].values

# Tidy
del data_frame

# clean strings
review_text = [data_helper.clean_str(x) for x in review_text]

# Assign unique word ids to each word in a corpus
max_sentence_length = 100
vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sentence_length)

word_ids = list(vocab_processor.fit_transform(review_text))

# Total number of unique words
vocabulary_size = len(vocab_processor.vocabulary_)

# Convert np arrays to lists to support popping of 0 values
word_ids = [i.tolist() for i in word_ids]

# Remove trailing 0s for sentences not 100 words long
# TIL: I tried benchmarking np.trim_zeros here to avoid the type conversion,
# but discovered that it is quite expensive

for sentence in word_ids:
    try:
        while sentence[-1] == 0:
            sentence.pop()
    except:
        pass


word_ids = filter(None, word_ids)

word_ids = np.array(list(word_ids))

window = 2

# inputs
pivot_words = []

# outputs
target_words = []

sentence_count = word_ids.shape[0]
for sentence in range(sentence_count):
    possible_pivots = word_ids[sentence][window:-window]

    for i in range(len(possible_pivots)):
        pivot = possible_pivots[i]
        targets = np.array([])
        # Example: slice elements [0, 2)
        neg_target = word_ids[sentence][i : window + i]

        # Example: slice elements [3, 5)
        pos_target = word_ids[sentence][i + window + 1: i + window + window + 1]

        targets = np.append(targets, [neg_target, pos_target]).flatten().tolist()

        for target in range(window*2):
            pivot_words.append(pivot)
            target_words.append(targets[target])


optimizer, loss, x, y, sess = build_word2vec(vocabulary_size)
