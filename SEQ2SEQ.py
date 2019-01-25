#-*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
import nn_utils


class SEQ2SEQ(object):

    def __init__(self, session, options, mode):
        super(SEQ2SEQ, self).__init__()

        self.session = session
        self.options = options
        self.mode = mode
        self.build_graph()

    def __del__(self):
        self.session.close()
        print("TensorFlow session is closed.")

    def build_graph(self):
        print("Building the TensorFlow graph...")
        opts = self.options

        self._create_placeholder()
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
            initializer=tf.random_uniform([opts.vocabulary_size, opts.embedding_size], -0.1, 0.1))
        enc_x_embed = tf.nn.embedding_lookup(embeddings, self.enc_x)
        dec_x_embed = tf.nn.embedding_lookup(embeddings, self.dec_x)

        # The encoder RNN.
        with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
            # Define the encoder cell.
            enc_gru_cell_fw = nn_utils.create_rnn_cell(
                opts.num_hidden_units, dropout=opts.dropout, mode=self.mode)
            enc_gru_cell_bw = nn_utils.create_rnn_cell(
                opts.num_hidden_units, dropout=opts.dropout, mode=self.mode)
            # Perform encoding.
            _, encode_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=enc_gru_cell_fw,
                cell_bw=enc_gru_cell_bw,
                inputs=enc_x_embed,
                sequence_length=self.enc_x_lens,
                dtype=tf.float32)
            encode_state = tf.concat(encode_state, axis=1)
            encode_state = tf.layers.dense(encode_state, opts.num_hidden_units)

        # The decoder RNN.
        with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
            # Define the decoder cell and the output layer.
            dec_gru_cell = nn_utils.create_rnn_cell(opts.num_hidden_units)
            output_layer = tf.layers.Dense(
                units=opts.vocabulary_size,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

            if self.mode == "train":
                sampling_prob = tf.constant(0.4, dtype=tf.float32)
                train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=dec_x_embed,
                    sequence_length=self.dec_x_lens,
                    embedding=embeddings,
                    sampling_probability=sampling_prob,
                    name="train_helper")
                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_gru_cell,
                    helper=train_helper,
                    initial_state=encode_state,
                    output_layer=output_layer)
                train_dec_outputs_, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    impute_finished=True,
                    maximum_iterations=opts.max_utterance_len)
                train_dec_outputs = tf.identity(train_dec_outputs_.rnn_output)
                max_target_length = tf.reduce_max(self.dec_x_lens)
                mask = tf.sequence_mask(
                    self.dec_x_lens, 
                    maxlen=max_target_length,
                    dtype=tf.float32)
                targets = tf.slice(self.dec_y, [0, 0], 
                    [opts.batch_size, max_target_length]) 
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    logits=train_dec_outputs,
                    targets=targets,
                    weights=mask) 
     
                self.optimizer = tf.train.AdamOptimizer(opts.learning_rate).minimize(self.loss)

            if self.mode == "predict":
                # Define the beam search decoder 
                start_tokens = tf.ones([opts.batch_size], dtype=tf.int32) * opts.go_index
                if opts.use_beam_search:
                    tiled_encoder_state = tf.contrib.seq2seq.tile_batch(
                        encode_state, multiplier=opts.beam_width)
                    predict_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=dec_gru_cell,
                        embedding=embeddings,
                        start_tokens=start_tokens,
                        end_token=opts.eos_index,
                        initial_state=tiled_encoder_state,
                        beam_width=opts.beam_width,
                        output_layer=output_layer)
                    self.predict_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=predict_decoder,
                        maximum_iterations=opts.max_utterance_len)
                else:
                    decoding_helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs=dec_x_embed,
                        sequence_length=self.dec_x_lens)
                    predict_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=dec_gru_cell,
                        helper=decoding_helper,
                        initial_state=encode_state,
                        output_layer=output_layer)

                    # Perform predict decoding.
                    max_target_length = tf.reduce_max(self.dec_x_lens)
                    self.predict_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=predict_decoder,
                        maximum_iterations=max_target_length)
                
                    # Compute evaulation loss.
                    mask = tf.sequence_mask(
                        self.dec_x_lens,
                        maxlen=max_target_length,
                        dtype=tf.float32)
                    targets = tf.slice(self.dec_y, [0, 0], 
                        [opts.batch_size, max_target_length])
                    predict_outputs = tf.identity(self.predict_dec_outputs.rnn_output)
                    self.loss = tf.contrib.seq2seq.sequence_loss(
                        logits=predict_outputs,
                        targets=targets,
                        weights=mask)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def _create_placeholder(self):
        opts = self.options
        self.enc_x = tf.placeholder(
            tf.int32, shape=[opts.batch_size, None])
        self.dec_x = tf.placeholder(
            tf.int32, shape=[opts.batch_size, None])
        self.dec_y = tf.placeholder(
            tf.int32, shape=[opts.batch_size, None])

        self.enc_x_lens = tf.placeholder(
            tf.int32, shape=[opts.batch_size])
        self.dec_x_lens = tf.placeholder(
            tf.int32, shape=[opts.batch_size])

    def init_tf_vars(self):
        self.session.run(self.init)
        print("TensorFlow variables initialized.")

    def train_step(self, enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens):
        feed_dict = {
            self.enc_x: enc_x,
            self.dec_x: dec_x,
            self.dec_y: dec_y,
            self.enc_x_lens: enc_x_lens,
            self.dec_x_lens: dec_x_lens}
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss
       
    def predict_step(self, enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens):
        feed_dict = {
            self.enc_x: enc_x,
            self.dec_x: dec_x, self.dec_y: dec_y,
            self.enc_x_lens: enc_x_lens,
            self.dec_x_lens: dec_x_lens}
        loss = 0.0
        if self.options.use_beam_search:
            dec_output = self.session.run(self.predict_dec_outputs, feed_dict=feed_dict)
            predicts = dec_output.predicted_ids
        else:
            dec_output, loss = self.session.run(
                [self.predict_dec_outputs, self.loss], feed_dict=feed_dict)
            predicts = dec_output.sample_id
        return predicts, loss

    def save(self, save_path):
        print("Saving the trained model...")
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print("Restoring from a pre-trained model...")
        self.saver.restore(self.session, restore_path)
