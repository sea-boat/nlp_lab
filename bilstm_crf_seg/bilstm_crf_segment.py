import tensorflow as tf
from bilstm_crf_seg import data_helper
from bilstm_crf_seg.util import get_logger
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import os
import time
import sys

embedding_dim = 100
hidden_dim = 128
clip_grad = 5
epoch_num = 100
batch_size = 128
lr = 0.001
dropout_keep_prob = 0.5

restore_model_path = os.path.join('.', "model_save", "1525597496/checkpoints")


label_num = len(data_helper.tag2label)
vocab_index_dict, index_vocab_dict, vocab_size = data_helper.load_vocab('vocab.json')

embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
embeddings = np.float32(embedding_mat)

train_data = data_helper.read_train_data()

test_data = train_data[0:1]


def train():
    timestamp = str(int(time.time()))
    output_path = os.path.join('.', "model_save", timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    result_path = os.path.join(output_path, "results")
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    logger = get_logger(log_path)
    graph = tf.Graph()
    with graph.as_default():
        word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=True, name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=word_ids, name="word_embeddings")
        word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)

        cell_fw = LSTMCell(hidden_dim)
        cell_bw = LSTMCell(hidden_dim)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                            inputs=word_embeddings,
                                                                            sequence_length=sequence_lengths,
                                                                            dtype=tf.float32)
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        output = tf.nn.dropout(output, dropout_pl)

        W = tf.get_variable(name="W", shape=[2 * hidden_dim, label_num],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name="b", shape=[label_num], initializer=tf.zeros_initializer(), dtype=tf.float32)
        s = tf.shape(output)
        output = tf.reshape(output, [-1, 2 * hidden_dim])
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, s[1], label_num])

        labels_softmax_ = tf.argmax(logits, axis=-1)
        labels_softmax_ = tf.cast(labels_softmax_, tf.int32)

        log_likelihood, transition_params = crf_log_likelihood(inputs=logits, tag_indices=labels,
                                                               sequence_lengths=sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", loss)

        with tf.variable_scope("train_step"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=lr_pl)
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
            train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(summary_path, sess.graph)
        for epoch in range(epoch_num):
            num_batches = (len(train_data) + batch_size - 1) // batch_size

            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            batches = data_helper.batch_yield(train_data, batch_size, vocab_index_dict, data_helper.tag2label)
            for step, (seqs, labs) in enumerate(batches):
                sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                step_num = epoch * num_batches + step + 1
                w_ids, seq_len_list = data_helper.pad_sequences(seqs, pad_mark=0)
                labels_, _ = data_helper.pad_sequences(labs, pad_mark=0)
                feed_dict = {word_ids: w_ids, sequence_lengths: seq_len_list, labels: labels_, lr_pl: lr,
                             dropout_pl: dropout_keep_prob}
                _, loss_train, summary, step_num_ = sess.run([train_op, loss, merged, global_step],
                                                             feed_dict=feed_dict)
                if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                    logger.info(
                        '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                    loss_train, step_num))
                file_writer.add_summary(summary, step_num)
                if step + 1 == num_batches:
                    saver.save(sess, model_path, global_step=step_num)


def predict():
    graph = tf.Graph()
    with graph.as_default():
        word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=True, name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=word_ids, name="word_embeddings")
        word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)

        cell_fw = LSTMCell(hidden_dim)
        cell_bw = LSTMCell(hidden_dim)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                            inputs=word_embeddings,
                                                                            sequence_length=sequence_lengths,
                                                                            dtype=tf.float32)
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        output = tf.nn.dropout(output, dropout_pl)

        W = tf.get_variable(name="W", shape=[2 * hidden_dim, label_num],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name="b", shape=[label_num], initializer=tf.zeros_initializer(), dtype=tf.float32)
        s = tf.shape(output)
        output = tf.reshape(output, [-1, 2 * hidden_dim])
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, s[1], label_num])

        labels_softmax_ = tf.argmax(logits, axis=-1)
        labels_softmax_ = tf.cast(labels_softmax_, tf.int32)

        log_likelihood, transition_params = crf_log_likelihood(inputs=logits, tag_indices=labels,
                                                               sequence_lengths=sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", loss)

        with tf.variable_scope("train_step"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=lr_pl)
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
            train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())
    with tf.Session(graph=graph) as sess:
        module_file = tf.train.latest_checkpoint(restore_model_path)
        saver.restore(sess, module_file)
        sent = '我是中国人'
        # sent = '深圳市宝安区'
        sent = list(sent.strip())
        sent_data = [(sent, ['U'] * len(sent))]

        label_list = []
        for seqs, labels in data_helper.batch_yield(sent_data, batch_size, vocab_index_dict, data_helper.tag2label,
                                                    shuffle=False):
            word_ids_, seq_len_list = data_helper.pad_sequences(seqs, pad_mark=0)
            feed_dict = {word_ids: word_ids_, sequence_lengths: seq_len_list, dropout_pl: 1}
            logits, transition_params = sess.run([logits, transition_params],
                                                 feed_dict=feed_dict)
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
        print(label_list)


if __name__ == '__main__':
    # train()
    predict()
