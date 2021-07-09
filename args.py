import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='Cluster the movie reviews')
    parser.add_argument('mode', nargs='+', default='load',help='select save or load vocab')
    return parser.parse_args()