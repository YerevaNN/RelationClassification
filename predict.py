from __future__ import print_function

import argparse
import json
import os

import numpy as np
from keras.models import load_model

from layers.decaying_dropout import DecayingDropout
from model import Classifier
from optimizers.l2optimizer import L2Optimizer
from preprocess import BioNLPPreprocessor


def predict(model, p, h, chars_per_word, preprocessor,
            input_path, output_path, mappings_dir,
            batch_size=70,
            include_word_vectors=True, include_chars=True,
            include_syntactical_features=True, include_exact_match=True):

    # Init mappings of the preprocessor
    preprocessor.load_mappings(word_mapping_file=os.path.join(mappings_dir, 'word_to_id.json'),
                               char_mapping_file=os.path.join(mappings_dir, 'car_to_id.json'),
                               part_of_speech_mapping_file=os.path.join(mappings_dir, 'part_of_speech_to_id.json'))

    with open(input_path, 'r') as f:
        data = json.load(f)

    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start: batch_start + batch_size]
        data_input = preprocessor.parse(data=batch,
                                        max_words_p=p,
                                        max_words_h=h,
                                        chars_per_word=chars_per_word,
                                        include_word_vectors=include_word_vectors,
                                        include_chars=include_chars,
                                        include_syntactical_features=include_syntactical_features,
                                        include_exact_match=include_exact_match)
        data_input = data_input[:-1]    # Last axis contains real labels
        model_outputs = model.predict(data_input)
        predictions = np.argmax(model_outputs, axis=1)
        for sample, prediction in zip(batch, predictions):
            sample['prediction'] = int(prediction)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=32,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=4,          help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--batch_size',     default=70,         help='Batch size while making predictions', type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    parser.add_argument('--dataset',        default='bionlp',   help='Which preprocessor to use',           type=str)
    parser.add_argument('--mappings_dir',   default='mappings', help='Path to mappings',                    type=str)
    parser.add_argument('--word_vec_load_path', default=None,   help='Path to load word vectors',           type=str)
    parser.add_argument('--normalize_word_vectors',      action='store_true')
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    parser.add_argument('--input',          default='data/bionlp_test_data.json',                           type=str)
    parser.add_argument('--output',         default='data/out_bionlp_test_data.json',                       type=str)
    parser.add_argument('--model',          default='models/best_model.hdf5',                               type=str)
    args = parser.parse_args()

    if args.dataset == 'bionlp':
        snli_preprocessor = BioNLPPreprocessor()
        # path = get_snli_file_path()
        model = load_model(args.model, custom_objects={'DIIN': Classifier,
                                                       'DecayingDropout': DecayingDropout,
                                                       'L2Optimizer': L2Optimizer})

        predict(model=model,
                p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                preprocessor=snli_preprocessor,
                input_path=args.input,
                output_path=args.output,
                mappings_dir=args.mappings_dir,
                batch_size=args.batch_size,
                include_word_vectors=not args.omit_word_vectors,
                include_chars=not args.omit_chars,
                include_syntactical_features=not args.omit_syntactical_features,
                include_exact_match=not args.omit_exact_match)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')


if __name__ == '__main__':
    main()
