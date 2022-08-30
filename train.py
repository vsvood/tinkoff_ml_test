"""This module is responsible for preparing data from raw text files, learning
n-gram model and dumping it to file"""
import re
import os
import pickle
import argparse
from n_gram_model import NGramModel


def extract_tokens(text: str) -> list[str]:
    """List all the word tokens in text."""
    return re.findall(r'\w+|\.', text.lower())


def read(input_dir: str) -> list[str]:
    """Read all files in input_dir and returns all data as list of lines"""
    (dir_path, _, filepaths) = next(os.walk(input_dir))
    filepaths = [os.path.join(dir_path, filepath) for filepath in filepaths]
    lines = []
    for filepath in filepaths:
        with open(filepath, 'r') as data_file:
            lines.extend(data_file.readlines())
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', metavar='<path>', dest='input_dir',
                        help='Path to data dir. Data should be stored as normal text files')
    parser.add_argument('--model', '-m', required=True, metavar='<path>', dest='model_path',
                        help='Path to file where to save model')
    args = parser.parse_args()

    if args.input_dir:
        raw_data = read(args.input_dir)
    else:
        raw_data = [input()]

    data = tuple(extract_tokens(''.join(raw_data)))
    model = NGramModel(2)
    model.fit(data)

    with open(args.model_path, 'wb') as file:
        pickle.dump(model, file)
