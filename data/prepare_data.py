#-*- coding: utf-8 -*- 


import sys
import nltk
import random
import itertools
import pandas as pd

def convert_install_to_numetric(s):
    t = s[:-1].replace(",", "")
    ret = int(t) if t.isdigit() else 0
    return ret

def handle_names(name):
    name = nltk.word_tokenize(name.lower())
    return name

def get_noun_and_verb(text):
    tags = nltk.pos_tag(text)
    keep_words = [w for w, tag in tags if "V" in tag or "N" in tag]
    return keep_words

def save_data(corpus, filepath):
    with open(filepath, "w", encoding="utf-8") as fw:
        for row in corpus:
            fw.write("\t".join(row) + "\n")

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

input_file = sys.argv[1]
data = pd.read_csv(input_file)

data["Installs"] = data["Installs"].apply(convert_install_to_numetric)
data = data[data["Installs"] >= 5000]
print (len(data))

data["App"] = data["App"].apply(handle_names)
data["Keywords"] = data["App"].apply(get_noun_and_verb)

corpus = []
data_items = list(data.iterrows())
random.shuffle(data_items)
for index, row in data_items:
    keywords = []
    for num in range(1, 5):
        keywords.extend(itertools.permutations(row["Keywords"], num))
    for keyword in keywords:
        corpus.append([" ".join(keyword), " ".join(row["App"])])

train_corpus, dev_corpus, test_corpus = corpus[:-2000], corpus[-2000:-1000], corpus[-1000:]
random.shuffle(train_corpus)
random.shuffle(dev_corpus)
random.shuffle(test_corpus)
save_data(train_corpus, "train.txt")
save_data(dev_corpus, "dev.txt")
save_data(test_corpus, "test.txt")

data["App"] = data["App"].apply(lambda t: " ".join(t))
data.to_csv("app_info.csv", encoding="utf-8", index=False)
