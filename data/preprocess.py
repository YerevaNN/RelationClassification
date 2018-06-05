from __future__ import print_function

import json

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
try:                import cPickle as pickle
except ImportError: import _pickle as pickle


def pad(x, max_len):
    if len(x) <= max_len:
        pad_width = ((0, max_len - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:max_len]
    return np.array(res)


class BasePreprocessor(object):
    def __init__(self, word_mapping=None, char_mapping=None,
                 part_of_speech_mapping=None, omit_labels=None,
                 max_words=(33, 20), chars_per_word=13,
                 include_word_vectors=True, include_chars=True, include_pos_tags=True,
                 include_amr_path=False, include_sdg_path=False, include_interaction_tuple=False):
        self.word_mapping = word_mapping
        self.char_mapping = char_mapping
        self.part_of_speech_mapping = part_of_speech_mapping
        self.omit_labels = omit_labels if omit_labels is not None else set()

        self.max_words = max_words
        self.chars_per_word = chars_per_word

        self.include_word_vectors = include_word_vectors
        self.include_chars = include_chars
        self.include_pos_tags = include_pos_tags

        self.include_interaction_tuple = include_interaction_tuple
        self.include_amr_path = include_amr_path
        self.include_sdg_path = include_sdg_path

    @staticmethod
    def load_data(file_path):
        with open(file_path) as f:
            lines = f.readlines()
            text = '[' + ','.join(lines) + ']'
            return json.loads(text)

    def get_words_with_part_of_speech(self, sample, sentence):
        raise NotImplementedError

    def get_sentences(self, sample):
        raise NotImplementedError

    def get_all_words_with_parts_of_speech(self, file_paths):
        all_words = []
        all_parts_of_speech = []
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)

            for sample in tqdm(data):
                sentences = self.get_sentences(sample)
                for sentence in sentences:
                    words, part_of_speech = self.get_words_with_part_of_speech(sample, sentence)
                    all_words += words
                    all_parts_of_speech += part_of_speech

        return all_words, all_parts_of_speech

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res

    def parse_sentence(self, sample, sentence, max_words, chars_per_word):
        words, parts_of_speech = self.get_words_with_part_of_speech(sample, sentence)

        word_ids = [self.word_mapping[word] for word in words]                          # Words
        pos_tag_ids = [self.part_of_speech_mapping[part] for part in parts_of_speech]   # Syntactical features
        char_ids = [[self.char_mapping[c] for c in word] for word in words]             # Chars
        char_ids = pad_sequences(char_ids, maxlen=chars_per_word, padding='post', truncating='post')

        res = []
        if self.include_word_vectors:   res.append(np.array(word_ids))
        if self.include_pos_tags:       res.append(np.array(pos_tag_ids))
        if self.include_chars:          res.append(np.array(pad(char_ids, max_words)))
        return tuple(res)

    def parse_one(self, sample, sentences, max_words, chars_per_word):
        """ :return: [ (word_ids, pos_tag_ids, char_ids) for sentence in sentences ] """
        return [self.parse_sentence(sample=sample, sentence=sentence,
                                    max_words=max_w, chars_per_word=chars_per_word)
                for sentence, max_w in zip(sentences, max_words)]

    def parse(self, data, verbose=False):
        """ :return: [word_ids..., pos_tag_ids..., char_ids...] """
        res = []
        for sample in tqdm(data) if verbose else data:
            if self.skip_sample(sample=sample):
                continue

            label = self.get_label(sample=sample)
            sentences = self.get_sentences(sample=sample)
            sample_inputs = self.parse_one(sample=sample, sentences=sentences,
                                           max_words=self.max_words, chars_per_word=self.chars_per_word)
            label = self.label_to_one_hot(label=label)

            sample_result = []
            for j in range(len(sample_inputs[0])):
                for i in range(len(sample_inputs)):
                    sample_result.append(sample_inputs[i][j])
            sample_result.append(label)
            res.append(sample_result)

        res = list(zip(*res))
        for i in range(len(res) - 1):
            res[i] = pad_sequences(res[i], maxlen=self.max_words[i % len(self.max_words)], padding='post', truncating='post')
        res[-1] = np.array(res[-1])
        return res

    def parse_file(self, input_file_path):
        data = self.load_data(input_file_path)
        return self.parse(data=data)

    def skip_sample(self, sample):
        label = self.get_label(sample=sample)
        if label in self.omit_labels:
            return True
        return False


class BioNLPPreprocessor(BasePreprocessor):

    def __init__(self, omit_interactions=None, include_single_interaction=True, **kwargs):
        self.valid_interactions = omit_interactions
        self.include_single_interaction = include_single_interaction
        super(BioNLPPreprocessor, self).__init__(**kwargs)

    @staticmethod
    def load_data(file_path):
        with open(file_path) as f:
            raw_data = json.loads(f.read().lower())  # case insensitive data
        res = []
        for key, value in raw_data.items():
            value.update({'id': key})
            res.append(value)
        return res

    def get_words_with_part_of_speech(self, sample, sentence):
        if sentence == self.get_sentences(sample)[0]:
            words = sample['tokenized_text'] if 'tokenized_text' in sample else sample['text'].split()
            pos_tags = sample['pos_tags'] if self.include_pos_tags else ['X'] * len(words)
        else:
            words = sentence.split(' ')
            pos_tags = ['X'] * len(words)
        return words, pos_tags

    def get_sentences(self, sample):
        text = ' '.join(sample['tokenized_text']) if 'tokenized_text' in sample else sample['text']
        interaction_tuple = sample['interaction_tuple']
        interaction_tuple = [item for item in interaction_tuple if item is not None]

        res = [text]
        if self.include_interaction_tuple:      res.append(' '.join(interaction_tuple))
        if self.include_amr_path:               res.append(self.get_amr_path(sample=sample))
        if self.include_sdg_path:               res.append(self.get_sdg_path(sample=sample))
        return tuple(res)

    @staticmethod
    def get_amr_path(sample):
        return sample['amr_path']

    @staticmethod
    def get_sdg_path(sample):
        return sample['sdg_path']

    def get_label(self, sample):
        return sample['label']

    def get_labels(self):
        return 0, 1

    def skip_sample(self, sample):
        interaction_tuple = sample['interaction_tuple']
        interaction_tuple = [item for item in interaction_tuple if item is not None]
        interaction_type = interaction_tuple[0]
        if self.valid_interactions is not None and interaction_type not in self.valid_interactions:     return True
        if not self.include_single_interaction and len(interaction_tuple) == 2:                         return True
        return False
