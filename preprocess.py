import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default="data/total.txt",
                    help = "path to the data file")
parser.add_argument("--max_utterance_len", type = int, default = 10,
                    help = "maximum length of utterance")
parser.add_argument("--vocabulary_size", type = int, default = 10000,
                    help = "vocabulary size")
parser.add_argument("--save_path", type = str, default = "data",
                    help = "the directory to save the data to")

max_context_length = 4

def read_data(file_path, max_utterance_len, max_text_len=None):
    texts = []
    frequencies = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip().split("\t")
            if max_text_len:
                text = text[-max_text_len:]
            text = [utterance.lower().split() for utterance in text]
            text = [utterance[:max_utterance_len] for utterance in text]
            for utterance in text:
                for token in utterance:
                    frequencies[token] += 1
            texts.append(text)
    return texts, frequencies

def construct_vocabulary(frequencies, vocabulary_size):
    tokens = sorted(frequencies, key = frequencies.get, reverse = True)
    tokens = ["<pad>", "<go>", "<eos>"] + tokens[:vocabulary_size]
    vocabulary = dict(zip(tokens, range(len(tokens))))
    vocabulary_reverse = dict(zip(range(len(tokens)), tokens))
    return vocabulary, vocabulary_reverse

def save_vocab(vocabulary, vocabulary_reverse, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open("{}/vocabulary.pickle".format(save_path), "wb") as file:
        pickle.dump(vocabulary, file)
    with open("{}/vocabulary_reverse.pickle".format(save_path), "wb") as file:
        pickle.dump(vocabulary_reverse, file)

def load_vocab(data_path):
    with open("{}/vocabulary.pickle".format(data_path), "rb") as file:
        vocabulary = pickle.load(file)
    with open("{}/vocabulary_reverse.pickle".format(data_path), "rb") as file:
        vocabulary_reverse = pickle.load(file)
    return vocabulary, vocabulary_reverse

def print_info(texts, frequencies):
    print("Total number of texts: {}".format(len(texts)))
    print("Total vocabulary size: {}".format(len(frequencies)))
    max_text_len = np.max([len(text) for text in texts])
    print("Max number of turns: {}".format(max_text_len))
    max_utterance_len = np.max([len(utterance) for text in texts for utterance in text])
    print("Max number of tokens: {}".format(max_utterance_len))

def convert_to_integer(texts, vocabulary):
    max_utterance_len = np.max([len(text[-1]) for text in texts])
    enc_x = np.zeros((len(texts), max_context_length + 2), dtype=np.int32)
    dec_x = np.zeros((len(texts), max_utterance_len + 1), dtype=np.int32)
    dec_y = np.zeros((len(texts), max_utterance_len + 1), dtype=np.int32)
    enc_x_lens = np.zeros((len(texts),), dtype=np.int32)
    dec_x_lens = np.zeros((len(texts),), dtype=np.int32)
    for i in range(len(texts)):
        context = []
        for utterance in texts[i][:-1]:
            context.extend(utterance)
        context = context[-max_context_length:]
        response = texts[i][-1]

        context = [w for w in context if w in vocabulary]
        response = [w for w in response if w in vocabulary]
        enc_x_ = ["<go>"] + context + ["<eos>"]
        dec_x_ = ["<go>"] + response
        dec_y_ = response + ["<eos>"]
        enc_x_lens[i] = len(enc_x_)
        dec_x_lens[i] = len(dec_x_)
        
        enc_x[i][:len(enc_x_)] = [vocabulary[w] for w in enc_x_]
        dec_x[i][:len(dec_x_)] = [vocabulary[w] for w in dec_x_]
        dec_y[i][:len(dec_y_)] = [vocabulary[w] for w in dec_y_]
        
    return enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens

if __name__ == "__main__":
    args = parser.parse_args()
    # Read the text text from the txt file.
    texts, frequencies = read_data(args.file_path, args.max_utterance_len)
    print_info(texts, frequencies)
    # Construct the vocabulary with the most frequent tokens.
    vocabulary, vocabulary_reverse = construct_vocabulary(frequencies, args.vocabulary_size)
    # Save the data to files.
    save_vocab(vocabulary, vocabulary_reverse, args.save_path)
