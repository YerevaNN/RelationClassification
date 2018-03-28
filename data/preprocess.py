from __future__ import print_function

import argparse
import json
import os

import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from data.mappings import WordVectors, CharToIdMapping, KeyToIdMapping
from util import get_word2vec_file_path, ChunkDataManager
try:                import cPickle as pickle
except ImportError: import _pickle as pickle


def pad(x, max_len):
    if len(x) <= max_len:
        pad_width = ((0, max_len - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:max_len]
    return np.array(res)


class BasePreprocessor(object):
    def __init__(self, word_mapping=None, char_mapping=None, part_of_speech_mapping=None, omit_labels=None,
                 max_words_p=33, max_words_h=20, chars_per_word=13,
                 include_word_vectors=True, include_chars=True,
                 include_syntactical_features=True, include_exact_match=True, include_amr_path=True):
        self.word_mapping = word_mapping
        self.char_mapping = char_mapping
        self.part_of_speech_mapping = part_of_speech_mapping
        self.omit_labels = omit_labels if omit_labels is not None else set()

        self.max_words_p = max_words_p
        self.max_words_h = max_words_h
        self.chars_per_word = chars_per_word

        self.include_word_vectors = include_word_vectors
        self.include_chars = include_chars
        self.include_syntactical_features = include_syntactical_features
        self.include_exact_match = include_exact_match
        self.include_amr_path = include_amr_path

    @staticmethod
    def load_data(file_path):
        """
        Load jsonl file by default
        """
        with open(file_path) as f:
            lines = f.readlines()
            text = '[' + ','.join(lines) + ']'
            return json.loads(text)

    def get_words_with_part_of_speech(self, sentence):
        """
        :return: words, parts_of_speech
        """
        raise NotImplementedError

    def get_sentences(self, sample):
        """
        :param sample: sample from data
        :return: premise, hypothesis
        """
        raise NotImplementedError

    def get_all_words_with_parts_of_speech(self, file_paths):
        """
        :param file_paths: paths to files where the data is stored
        :return: words, parts_of_speech
        """
        all_words = []
        all_parts_of_speech = []
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)

            for sample in tqdm(data):
                premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(premise)
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(hypothesis)
                all_words           += premise_words  + hypothesis_words
                all_parts_of_speech += premise_speech + hypothesis_speech

        return all_words, all_parts_of_speech

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def get_amr_path(self, sample):
        return None

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res

    def parse_sentence(self, sentence, max_words, chars_per_word):
        # Words
        words, parts_of_speech = self.get_words_with_part_of_speech(sentence)
        word_ids = [self.word_mapping[word] for word in words]

        # Syntactical features
        syntactical_features = [self.part_of_speech_mapping[part] for part in parts_of_speech]
        syntactical_one_hot = np.eye(len(self.part_of_speech_mapping) + 2)[syntactical_features]  # Convert to 1-hot

        # Chars
        chars = [[self.char_mapping[c] for c in word] for word in words]
        chars = pad_sequences(chars, maxlen=chars_per_word, padding='post', truncating='post')

        return (words, parts_of_speech, np.array(word_ids, copy=False),
                syntactical_features, pad(syntactical_one_hot, max_words),
                pad(chars, max_words))

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        """
        :param premise: sentence
        :param hypothesis: sentence
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        (premise_words, premise_parts_of_speech, premise_word_ids,
         premise_syntactical_features, premise_syntactical_one_hot,
         premise_chars) = self.parse_sentence(sentence=premise, max_words=max_words_p, chars_per_word=chars_per_word)

        (hypothesis_words, hypothesis_parts_of_speech, hypothesis_word_ids,
         hypothesis_syntactical_features, hypothesis_syntactical_one_hot,
         hypothesis_chars) = self.parse_sentence(sentence=hypothesis, max_words=max_words_h, chars_per_word=chars_per_word)

        def calculate_exact_match(source_words, target_words):
            source_words = [word.lower() for word in source_words]
            target_words = [word.lower() for word in target_words]
            target_words = set(target_words)

            res = [(word in target_words) for word in source_words]
            return np.array(res)

        premise_exact_match    = calculate_exact_match(premise_words, hypothesis_words)
        hypothesis_exact_match = calculate_exact_match(hypothesis_words, premise_words)

        return (premise_word_ids, hypothesis_word_ids,
                premise_chars, hypothesis_chars,
                premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                premise_exact_match, hypothesis_exact_match)

    def parse(self, data, verbose=False):
        """
        :param data: data to parse
        :param verbose: to show progress or not
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        # res = [premise_word_ids, hypothesis_word_ids, premise_chars, hypothesis_chars,
        # premise_syntactical_one_hot, hypothesis_syntactical_one_hot, premise_exact_match, hypothesis_exact_match]
        res = [[], [], [], [], [], [], [], [], []]

        for sample in tqdm(data) if verbose else data:
            if self.skip_sample(sample=sample):
                continue
            label = self.get_label(sample=sample)
            premise, hypothesis = self.get_sentences(sample=sample)

            ''' Add AMR path to hypothesis '''
            if self.include_amr_path:
                amr_path = self.get_amr_path(sample)
                if amr_path != '':
                    hypothesis = amr_path
                else:
                    hypothesis = hypothesis.split(' ')
                    hypothesis = hypothesis[-2] + ' unknown ' + hypothesis[-1]

            sample_inputs = self.parse_one(premise, hypothesis,
                                           max_words_h=self.max_words_h, max_words_p=self.max_words_p,
                                           chars_per_word=self.chars_per_word)
            label = self.label_to_one_hot(label=label)

            sample_result = list(sample_inputs) + [label]
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=self.max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=self.max_words_h, padding='post', truncating='post', value=0.)  # input_word_h
        res[6] = pad_sequences(res[6], maxlen=self.max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=self.max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h

        # Determine which part of data we need to dump
        if not self.include_exact_match:             del res[6:8]  # Exact match feature
        if not self.include_syntactical_features:    del res[4:6]  # Syntactical POS tags
        if not self.include_chars:                   del res[2:4]  # Character features
        if not self.include_word_vectors:            del res[0:2]  # Word vectors
        return [np.array(item) for item in res]

    def parse_file(self, input_file_path):
        data = self.load_data(input_file_path)
        return self.parse(data=data)

    def skip_sample(self, sample):
        label = self.get_label(sample=sample)
        if label in self.omit_labels:
            return True
        return False


class BioNLPPreprocessor(BasePreprocessor):

    def __init__(self, omit_interactions=None, **kwargs):
        self.valid_interactions = omit_interactions
        super(BioNLPPreprocessor, self).__init__(**kwargs)

    @staticmethod
    def load_data(file_path):
        with open(file_path) as f:
            raw_data = json.load(f)
        res = []
        for key, value in raw_data.items():
            value.update({'id': key})
            res.append(value)
        return res

    def get_words_with_part_of_speech(self, sentence):
        words = sentence.split()  # nltk.word_tokenize(sentence)
        parts_of_speech = ['X'] * len(words)
        return words, parts_of_speech

    def get_sentences(self, sample):
        text = sample['text']
        interaction_tuple = sample['interaction_tuple']
        interaction_tuple = [item for item in interaction_tuple if item is not None]
        return text, ' '.join(interaction_tuple)

    def get_amr_path(self, sample):
        return sample['amr_path']

    def get_label(self, sample):
        return sample['label']

    def get_labels(self):
        return 0, 1

    def skip_sample(self, sample):
        interaction_tuple = sample['interaction_tuple']
        interaction_type = interaction_tuple[0]
        if self.valid_interactions is not None and interaction_type not in self.valid_interactions:
            return True
        return False


def preprocess(preprocessor, save_dir, data_paths, processor_save_path,
               normalize_word_vectors, max_loaded_word_vectors=None, word_vectors_load_path=None):

    all_words, all_parts_of_speech = preprocessor.get_all_words_with_parts_of_speech([data_path[1] for data_path in data_paths])

    ''' Mappings '''
    word_mapping = WordVectors()
    word_mapping.load(file_path=get_word2vec_file_path(word_vectors_load_path),
                      needed_words=set(all_words),
                      normalize=normalize_word_vectors,
                      max_words=max_loaded_word_vectors)

    char_mapping = CharToIdMapping(set(all_words))
    part_of_speech_mapping = KeyToIdMapping(keys=set(all_parts_of_speech))

    ''' Initialize preprocessor mappings '''
    preprocessor.word_mapping = word_mapping
    preprocessor.char_mapping = char_mapping
    preprocessor.part_of_speech_mapping = part_of_speech_mapping

    if processor_save_path is not None:
        with open(processor_save_path, 'wb') as f:
            pickle.dump(preprocessor, file=f)

    ''' Process and save the data '''
    for dataset, input_path in data_paths:
        data = preprocessor.parse_file(input_file_path=input_path)
        data_saver = ChunkDataManager(save_data_path=os.path.join(save_dir, dataset))
        data_saver.save(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=32,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=4,          help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    parser.add_argument('--save_dir',       default='data/',    help='Save directory of data',              type=str)
    parser.add_argument('--dataset',        default='bionlp',   help='Which preprocessor to use',           type=str)
    parser.add_argument('--word_vec_load_path', default=None,   help='Path to load word vectors',           type=str)
    parser.add_argument('--processor_save_path',    default='data/processor.pkl', help='Path to save vectors', type=str)
    parser.add_argument('--normalize_word_vectors',      action='store_true')
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    args = parser.parse_args()

    if args.dataset == 'bionlp':
        preprocessor = BioNLPPreprocessor(max_words_p=args.p,
                                          max_words_h=args.h,
                                          chars_per_word=args.chars_per_word,
                                          include_word_vectors=not args.omit_word_vectors,
                                          include_chars=not args.omit_chars,
                                          include_syntactical_features=not args.omit_syntactical_features,
                                          include_exact_match=not args.omit_exact_match)
        # path = get_snli_file_path()
        path = './data'
        train_path = os.path.join(path, 'bionlp_train_data.json')
        test_path  = os.path.join(path, 'bionlp_test_data.json')
        dev_path   = os.path.join(path, 'bionlp_valid_data.json')
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')

    preprocess(preprocessor=preprocessor,
               save_dir=args.save_dir,
               data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
               word_vectors_load_path=args.word_vec_load_path,
               normalize_word_vectors=args.normalize_word_vectors,
               processor_save_path=args.processor_save_path,
               max_loaded_word_vectors=args.max_word_vecs)


if __name__ == '__main__':
    main()
