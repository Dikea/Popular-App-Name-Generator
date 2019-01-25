#-*- coding: utf-8 -*-

import os
import time
import pickle
import random
import argparse
import datetime
import tensorflow as tf
import numpy as np
from SEQ2SEQ import SEQ2SEQ
from preprocess import read_data, load_vocab
from preprocess import convert_to_integer
from config import args, options


def run_epoch(model, examples):
    losses = []
    random.shuffle(examples)
    num_examples = len(examples)
    num_batches = num_examples // options.batch_size
    for batch in range(num_batches):
        s = batch * options.batch_size
        t = s + options.batch_size
        batch_examples = examples[s:t]
        enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens = (
            convert_to_integer(batch_examples, vocabulary))
        if model.mode == "train":
            loss = model.train_step(
                enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens)
        elif model.mode == "predict":
            _, loss = model.predict_step(
                enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens)
        losses.append(loss)
        if (batch + 1) % 100 == 0:
            print("[{}] batch={:04d}, loss={:.4f}".format(
                datetime.datetime.now(), batch + 1, loss))
    avg_loss = np.mean(losses)
    return avg_loss


if __name__ == "__main__":
    time_start = time.time()
    vocabulary, vocabulary_reverse = load_vocab(args.data_path)
    train_examples, _ = read_data(args.train_file, 
        args.max_utterance_len, args.max_example_len + 1)
    eval_examples, _ = read_data(args.eval_file, 
        args.max_utterance_len, args.max_example_len + 1)
    if not os.path.exists(args.root_path):
        os.makedirs(args.root_path)  

    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            train_model = SEQ2SEQ(session, options, "train")
        with tf.variable_scope("Model", reuse=True):
            eval_model = SEQ2SEQ(session, options, "predict") 

    if args.restore:    
        train_model.restore(os.path.join(args.root_path, args.restore_path))
    else:
        train_model.init_tf_vars()
 
    min_loss = np.inf
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(train_model, train_examples)
        eval_loss = run_epoch(eval_model, eval_examples)
        print ("Epoch={}, train_loss={:.4f}".format(epoch, train_loss))
        print ("Epoch={}, eval_loss={:.4f}".format(epoch, eval_loss))
        if eval_loss < min_loss:
            min_loss = eval_loss
            print ("Minimum loss reduced, save model, "
                   "min_loss={:.4f}".format(min_loss))
        train_model.save(options.save_path)

    print ("Time to train model is {:.4f} minutes.".format(
        (time.time() - time_start) / 60.0))
