#-*- coding: utf-8 -*-

import nltk


def corpus_bleu_score(references, hypotheses):
    references = [[r] for r in references]
    score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses) 
    return score
