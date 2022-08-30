"""This module is responsible for generating text
from given sample with specified model"""
import re
import pickle
import argparse


def extract_tokens(text: str) -> list[str]:
    """List all the word tokens in text."""
    return re.findall(r'\w+', text.lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, metavar='<path>', dest='model_path',
                        help='Path where to load model from')
    parser.add_argument('--prefix', '-p', default='', metavar='<str>', dest='prefix',
                        help='Prefix for generated phrase')
    parser.add_argument('--length', '-l', required=True, metavar='<int>', dest='length',
                        help='Length of generated phrase')
    args = parser.parse_args()

    with open(args.model_path, 'rb') as file:
        model = pickle.load(file)
    phrase = extract_tokens(args.prefix)
    for _ in range(int(args.length)):
        phrase.append(model.generate(tuple(phrase)))

    print(' '.join(phrase))
