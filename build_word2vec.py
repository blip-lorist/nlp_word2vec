import tensorflow as tf

def build_word2vec(vocabulary_size):
    # Pivot Worlds
    x = tf.placeholder(tf.int32, shape=[None,], name="x_pivot_idxs")

    # Target words
    y = tf.placeholder(tf.int32, shape=[None,], name="y_target_idxs")

    embedding_size = 128
    num_samples = 64
    learning_rate = 0.001

    # Make our word embedding matrix
    # This is a good description of embeddings:
    # https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
    # generate an embedding that is vocab size by embedding size,
    # using reals with a min value of -1 and max value of 1

    Embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
            name = "word_embedding")


    # Weights and biases and NCE Loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
        stddev = tf.sqrt(1/embedding_size), name = "nce_weights"))

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name = "nce_biases")

    pivot = tf.nn.embedding_lookup(Embedding, x, name = "word_embed_lookup")

    train_labels = tf.reshape(y, [tf.shape(y)[0], 1])

    # Computer loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases = nce_biases,
                                         labels = train_labels,
                                         inputs = pivot,
                                         num_sampled = num_samples,
                                         num_classes = vocabulary_size,
                                         num_true = 1))

    # Create optimizer
    optimizer = tf.contrib.layers.optimize_loss(loss,
                                                tf.train.get_global_step(),
                                                learning_rate,
                                                "Adam",
                                                clip_gradients = 5.0,
                                                name = "optimizer")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return optimizer, loss, x, y, sess