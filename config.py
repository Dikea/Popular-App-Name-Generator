#-*- coding: utf-8 -*-

import os
import argparse
from preprocess import load_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

corpus_path = {
    "GooglePlay": {
        "train": "data/train.txt",
        "dev": "data/dev.txt",
        "test": "data/test.txt"
        }
    }

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_name", type=str, default="GooglePlay", 
                    help="corpus name: GooglePlay")
parser.add_argument("--data_path", type=str, default="data",
                    help="the directory to the training data")
parser.add_argument("--app_info_path", type=str, default="data/app_info.csv",
                    help="the path to app info file")
parser.add_argument("--num_epochs", type=int, default=10,
                    help="the number of epochs to train the data")
parser.add_argument("--batch_size", type=int, default=128,
                    help="the batch size")
parser.add_argument("--max_utterance_len", type=int, default=35,
                    help="the max length of utterance")
parser.add_argument("--max_example_len", type=int, default=5,
                    help="the max length of example context")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="the learning rate")
parser.add_argument("--use_beam_search", type=bool, default=False,
                    help="whether to use beam search.")
parser.add_argument("--beam_width", type=int, default=10,
                    help="the beam width when decoding")
parser.add_argument("--dropout", type=float, default=0.25,
                    help="the value of dropout during training")
parser.add_argument("--embedding_size", type=int, default=128,
                    help="the size of word embeddings")
parser.add_argument("--num_hidden_layers", type=int, default=1,
                    help="the number of hidden layers")
parser.add_argument("--num_hidden_units", type=int, default=256,
                    help="the number of hidden units")
parser.add_argument("--root_path", type=str, default="Model",
                    help="root path to save model files")
parser.add_argument("--save_path", type=str, default="model/model.ckpt",
                    help="the path to save the trained model to")
parser.add_argument("--restore_path", type=str, default="model/model.ckpt",
                    help="the path to restore the trained model")
parser.add_argument("--restore", type=bool, default=False,
                    help="whether to restore from a trained model")

args = parser.parse_args()
args.train_file = corpus_path[args.corpus_name]["train"]
args.eval_file = corpus_path[args.corpus_name]["dev"]
args.test_file = corpus_path[args.corpus_name]["test"]
print (args)


# Set options.
class Options(object):
    """Parameters used by the SEQ2SEQ model."""
    def __init__(self, num_epochs, batch_size, learning_rate, beam_width, 
                 dropout, vocabulary_size, embedding_size, num_hidden_layers, 
                 num_hidden_units, use_beam_search,  max_example_len, 
                 max_utterance_len, go_index, eos_index, save_path):

        super(Options, self).__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beam_width = beam_width
        self.dropout = dropout
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.use_beam_search = use_beam_search
        self.max_example_len = max_example_len
        self.max_utterance_len = max_utterance_len
        self.go_index = go_index
        self.eos_index = eos_index
        self.save_path = save_path

vocabulary, vocabulary_reverse = load_vocab(args.data_path)
options = Options(
    num_epochs = args.num_epochs,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    use_beam_search = args.use_beam_search,
    beam_width = args.beam_width,
    dropout = args.dropout,
    vocabulary_size = len(vocabulary),
    embedding_size = args.embedding_size,
    num_hidden_layers = args.num_hidden_layers,
    num_hidden_units = args.num_hidden_units,
    max_example_len = args.max_example_len,
    max_utterance_len = args.max_utterance_len + 2,
    go_index = vocabulary["<go>"],
    eos_index = vocabulary["<eos>"],
    save_path = os.path.join(args.root_path, args.save_path))
