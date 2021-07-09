from preprocess import *
from args import *

if __name__ == '__main__':
    data_dir = 'dataset/movie_review/txt_sentoken/'
    pos = 'pos/'
    neg = 'neg/'
    filename = 'vocab.txt'
    train_size = 1800
    test_size = 200

    args = parse_arg()
    mode = args.mode[-1]

    if mode == 'save':
        print('Make the vocabulary and save on the vocab.txt')
        make_vocab_file(data_dir, pos, neg)
    elif mode == 'load':
        print('Load from vocab.txt')
        
        (X_train, y_train), (X_test, y_test) = make_dataset(filename, data_dir, pos, neg, train_size, test_size)

        n_words = X_test.shape[1]

        print(f'train dataset: X-{X_train.shape}, y-{y_train.shape}')
        print(f'test dataset: X-{X_test.shape}, y-{y_test.shape}')

        model = make_model(n_words)

        print(model)

        model.fit(X_train, y_train, epochs=50, verbose=2)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Accuracy: {acc*100}%')
    else:
        print(f'status: FAIL, content: mode selection')