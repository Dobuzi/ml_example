from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout

from string import punctuation
from os import listdir
from collections import Counter
from numpy import array, half

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
    return ''

def process_docs(directory, vocab, callback, is_train=True):
    lines = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        line = callback(path, vocab)
        lines.append(line)
    return lines

def make_tokens(vocab, min_occur):
    return [token for token, count in vocab.items() if count >= min_occur]

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def make_doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def make_vocab_file(data_dir, pos, neg, is_train):
    vocab = Counter()
    _ = process_docs(data_dir+pos, vocab, add_doc_to_vocab, is_train)
    _ = process_docs(data_dir+neg, vocab, add_doc_to_vocab, is_train)
    print(f'vocabulary length: {len(vocab)}')
    # print(vocab.most_common(50))

    min_occur = 2
    tokens = make_tokens(vocab, min_occur)
    print(f'token length: {len(tokens)}')

    save_list(tokens, 'vocab.txt')

def load_vocab_file(filename, data_dir, pos, neg, is_train):
    vocab = load_doc(filename)
    vocab = vocab.split()
    vocab = set(vocab)

    pos_lines = process_docs(data_dir+pos, vocab, make_doc_to_line, is_train)
    neg_lines = process_docs(data_dir+neg, vocab, make_doc_to_line, is_train)

    print(f'positive lines length: {len(pos_lines)}')
    print(f'negative lines length: {len(neg_lines)}')

    return neg_lines + pos_lines

def make_vector(tokenizer, docs):
    return tokenizer.texts_to_matrix(docs, mode='freq')

def make_y_vector(size):
    half_size = int(size/2)
    return array([0 for _ in range(half_size)] + [1 for _ in range(half_size)])

def make_dataset(filename, data_dir, pos, neg, train_size, test_size):
    docs_train = load_vocab_file(filename, data_dir, pos, neg, is_train=True)
    docs_test = load_vocab_file(filename,data_dir, pos, neg, is_train=False)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs_train)
    
    X_train = make_vector(tokenizer, docs_train)
    y_train = make_y_vector(train_size)

    X_test = make_vector(tokenizer, docs_test)
    y_test = make_y_vector(test_size)

    return (X_train, y_train), (X_test, y_test)

def make_model(n_words):
    model = Sequential()
    model.add(Input(shape=(n_words,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model