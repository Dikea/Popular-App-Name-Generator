#-*- coding: utf-8 -*-

import os 
import pickle
import argparse
import numpy as np
import tensorflow as tf
from SEQ2SEQ import SEQ2SEQ  
from preprocess import convert_to_integer, load_vocab 
from utils import corpus_bleu_score
from config import args, options


def read_test_data():
    with open(args.test_file, "r", encoding="utf-8") as f:
        examples, responses = [], []
        for line in f:
            example = line.split("\t")[-6:]
            example = [s.lower().split()[:args.max_utterance_len] for s in example]
            examples.append(example)
            responses.append(example[-1])
    return examples, responses

if __name__ == "__main__":  
    vocabulary, vocabulary_reverse = load_vocab(args.data_path)
    examples, responses = read_test_data()

    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model"):
            model = SEQ2SEQ(session, options, "predict")
    model.restore(os.path.join(args.root_path, args.restore_path))

    num_examples = len(examples) 
    num_batches = num_examples // options.batch_size
    predict_responses = []
    losses = []
    for batch in range(num_batches):
        s = batch * options.batch_size
        t = s + options.batch_size
        batch_examples = examples[s:t]
        batch_enc_x, batch_dec_x, batch_dec_y, batch_enc_x_lens, batch_dec_x_lens = (
            convert_to_integer(batch_examples, vocabulary))
        responses_, loss = model.predict_step(batch_enc_x, batch_dec_x, 
            batch_dec_y, batch_enc_x_lens, batch_dec_x_lens)
        for i in range(options.batch_size):
            ids = responses_[i].tolist()
            predict_response = []
            for idx in ids:
                if idx == vocabulary["<eos>"]:
                    break
                predict_response.append(vocabulary_reverse[idx])
            predict_responses.append(predict_response)
        if batch % 100 == 0: print ("idx={}, loss={:.4f}".format(batch, loss))
        losses.append(loss)
    bleu_score = corpus_bleu_score(responses[:len(predict_responses)], predict_responses) 
    print ("Perlexity={:.6f}".format(np.exp(np.mean(losses))))
    print ("Bleu={:.6f}".format(bleu_score * 100))
    
    references_file = os.path.join(args.root_path, "references.txt")
    predictions_file = os.path.join(args.root_path, "predictions.txt")
    with open(references_file, "w", encoding="utf-8") as f1, \
        open(predictions_file, "w", encoding="utf-8") as f2:
        for reference, prediction in zip(responses, predict_responses):
            reference = " ".join(reference)
            prediction = " ".join(prediction)
            f1.write(reference + "\n")
            f2.write(prediction + "\n")
