#-*- coding: utf-8 -*-

import os 
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
import tensorflow as tf
from SEQ2SEQ import SEQ2SEQ  
from preprocess import convert_to_integer, load_vocab 
from utils import corpus_bleu_score
from config import args, options


class Inference(object):

    def __init__(self):
        self.vocabulary, self.vocabulary_reverse = load_vocab(args.data_path)
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model"):
                self.model = SEQ2SEQ(session, options, "predict")
        self.model.restore(os.path.join(args.root_path, args.restore_path))


    def do_inference(self, keywords):
        enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens = convert_to_integer(
            [[keywords, []]], self.vocabulary) 
        result, _ = self.model.predict_step(
            enc_x, dec_x, dec_y, enc_x_lens, dec_x_lens) 
        result = result[0] 
        num = result.shape[-1]
        predicts = []
        for i in range(num):
            predict = []
            for idx in result[:, i].tolist():
                if idx == self.vocabulary["<eos>"]:
                    break
                predict.append(self.vocabulary_reverse[idx])
            uniq_predict = []
            predict_len = len(predict)
            for i in range(predict_len):
                if i > 0 and predict[i] == predict[i - 1]:
                    continue
                uniq_predict.append(predict[i])
            if uniq_predict:
                predicts.append(uniq_predict)
        return predicts


class AppSearch(object):
    
    def __init__(self):
        self.app_infos = pd.read_csv(args.app_info_path)
        app_names = self.app_infos["App"].tolist()
        app_names = [s.split() for s in app_names]

        self.dictionary = corpora.Dictionary(app_names)
        corpus = [self.dictionary.doc2bow(s) for s in app_names]
        self.tfidf_model = models.TfidfModel(corpus)
        corpus_mm = self.tfidf_model[corpus]
        self.tfidf_index = similarities.MatrixSimilarity(corpus_mm)
        print ("Build tfidf model done.")

    def _text2vec(self, text):
        bow = self.dictionary.doc2bow(text)
        return self.tfidf_model[bow]

    def get_most_similar_app(self, query):
        vec = self._text2vec(query)
        sims = self.tfidf_index[vec]
        sims_sorted = sorted(list(enumerate(sims)), 
            key=lambda item: item[1], reverse=True)
        idx = 0
        while True:
            index = sims_sorted[idx][0]
            result = self.app_infos.iloc[index].to_dict()
            if result["App"].split() != query:
                break
            if idx > 10:
                break
            idx += 1
        return result


class InferenceApiHanler(object):
    
    @classmethod
    def init(cls):
        cls.inference_inst = Inference() 
        cls.app_search_inst = AppSearch()
        print("Enable inference model done.")

    @classmethod
    def predict_app_name(cls, params):
        keywords = params["query"].strip().split("|")
        keywords = [w.lower() for w in keywords]
        predicts = cls.inference_inst.do_inference(keywords)
        sim_app_infos = [cls.app_search_inst.get_most_similar_app(name) for name in predicts] 
        names = [" ".join(p) for p in predicts]

        rsp = []
        for name, info in zip(names, sim_app_infos):
            info["recommand_name"] = name
            for key, value in info.items():
                info[key] = str(value)
            rsp.append(info)

        return (200, rsp)


if __name__ == "__main__":
    InferenceApiHanler.init()
    keywords = "classical music for baby".split()
    result = InferenceApiHanler.predict_app_name({"query": "|".join(keywords)})
    print ("Input={}, Output={}".format(keywords, result))
