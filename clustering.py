from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from collections import Counter

def load_doc(filename):
    f = open(filename, 'r')
    doc  = f.read()
    f.close()
    return doc

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def process_docs(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)

if __name__ == '__main__':
    data_dir = 'dataset/movie_review/txt_sentoken/'
    pos = 'pos/'
    neg = 'neg/'

    vocab = Counter()
    
    process_docs(data_dir+pos, vocab)
    process_docs(data_dir+neg, vocab)

    print(len(vocab))
    print(vocab.most_common(50))
