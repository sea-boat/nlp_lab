import tensorflow as tf
import numpy as np
import datetime
from cnn_text_classify import data_helper

sequence_length = 0
classes_num = 3
vocabulary_size = 0
embedding_size = 100
l2_lambda = 0.01
batch_size = 50
epochs = 1000
drop_keep_prob = 0.5
filters_height = [2, 3, 4]
filter_num_per_height = [100, 100, 100]

print("Loading data...")
train_input, train_label, vocabulary, vocabulary_inv = data_helper.load_data()
vocabulary_size = len(vocabulary)
print(train_input, train_label, vocabulary, vocabulary_inv)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(train_label)))
x_shuffled = train_input[shuffle_indices]
y_shuffled = train_label[shuffle_indices]
sequence_length = train_input.shape[1]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Sequnence Length: {:d}".format(sequence_length))

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, [None, sequence_length])
    train_labels = tf.placeholder(tf.float32, [None, classes_num])
    keep_prob = tf.placeholder(tf.float32)
    l2_loss = tf.constant(0.0)

    with tf.device('/cpu:0'):
        # embedding layer
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        conv_inputs = tf.expand_dims(embed, -1)

    features_pooled = []
    for filter_height, filter_num in zip(filters_height, filter_num_per_height):
        # conv layer
        conv_filter = tf.Variable(tf.truncated_normal([filter_height, embedding_size, 1, filter_num], stddev=0.1))
        conv = tf.nn.conv2d(conv_inputs, conv_filter, strides=[1, 1, 1, 1], padding="VALID")
        bias = tf.Variable(tf.constant(0.1, shape=[filter_num]))
        feature_map = tf.nn.relu(tf.nn.bias_add(conv, bias))
        # pooling layer
        feature_pooled = tf.nn.max_pool(feature_map, ksize=[1, sequence_length - filter_height + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID')
        features_pooled.append(feature_pooled)

    filter_num_total = sum(filter_num_per_height)
    # fully connected layer
    features_pooled_flat = tf.reshape(tf.concat(features_pooled, 3), [-1, filter_num_total])
    features_pooled_flat_drop = tf.nn.dropout(features_pooled_flat, keep_prob)

    W = tf.get_variable("W", shape=[filter_num_total, classes_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[classes_num]))
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(features_pooled_flat_drop, W, b)

    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=train_labels)
    loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

    predictions = tf.argmax(scores, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(train_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

with tf.Session(graph=graph) as sess:
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(loss, aggregation_method=2)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)

    sess.run(tf.initialize_all_variables())
    batches = data_helper.get_batch(zip(x_shuffled, y_shuffled), batch_size, epochs)
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        feed_dict = {train_inputs: x_batch, train_labels: y_batch, keep_prob: drop_keep_prob}
        _, step, _loss, _accuracy = sess.run([train_op, global_step, loss, accuracy], feed_dict)
        time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, _loss, _accuracy))
