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
                 max_words_p=33, max_words_h=20, chars_per_word=13,
                 include_word_vectors=True, include_chars=True,
                 include_pos_tags=True, include_exact_match=True,
                 include_amr_path=False, include_sdg_path=False):
        self.word_mapping = word_mapping
        self.char_mapping = char_mapping
        self.part_of_speech_mapping = part_of_speech_mapping
        self.omit_labels = omit_labels if omit_labels is not None else set()

        self.max_words_p = max_words_p
        self.max_words_h = max_words_h
        self.chars_per_word = chars_per_word

        self.include_word_vectors = include_word_vectors
        self.include_chars = include_chars
        self.include_pos_tags = include_pos_tags
        self.include_exact_match = include_exact_match
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
                premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(sample, premise)
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(sample, hypothesis)
                all_words           += premise_words  + hypothesis_words
                all_parts_of_speech += premise_speech + hypothesis_speech

        return all_words, all_parts_of_speech

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def get_amr_path(self, sample):
        return None

    def get_sdg_path(self, sample):
        return None

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res

    def parse_sentence(self, sample, sentence, max_words, chars_per_word):
        # Words
        words, parts_of_speech = self.get_words_with_part_of_speech(sample, sentence)
        word_ids = [self.word_mapping[word] for word in words]

        # Syntactical features
        pos_tag_ids = [self.part_of_speech_mapping[part] for part in parts_of_speech]

        # Chars
        chars = [[self.char_mapping[c] for c in word] for word in words]
        chars = pad_sequences(chars, maxlen=chars_per_word, padding='post', truncating='post')

        return (words, parts_of_speech, np.array(word_ids, copy=False),
                pos_tag_ids,
                pad(chars, max_words))

    def parse_one(self, sample, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        (premise_words, premise_parts_of_speech, premise_word_ids,
         premise_syntactical_ids, premise_chars) = self.parse_sentence(sentence=premise, sample=sample, max_words=max_words_p, chars_per_word=chars_per_word)

        (hypothesis_words, hypothesis_parts_of_speech, hypothesis_word_ids,
         hypothesis_syntactical_ids, hypothesis_chars) = self.parse_sentence(sentence=hypothesis, sample=sample, max_words=max_words_h, chars_per_word=chars_per_word)

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
                premise_syntactical_ids, hypothesis_syntactical_ids,
                premise_exact_match, hypothesis_exact_match)

    def parse(self, data, verbose=False):
        # res = [premise_word_ids, hypothesis_word_ids, premise_chars, hypothesis_chars,
        # premise_pos_tag_ids, hypothesis_pos_tag_ids, premise_exact_match, hypothesis_exact_match]
        res = [[], [], [], [], [], [], [], [], []]

        for sample in tqdm(data) if verbose else data:
            if self.skip_sample(sample=sample):
                continue
            label = self.get_label(sample=sample)
            premise, hypothesis = self.get_sentences(sample=sample)

            if self.include_amr_path:
                ''' Add AMR path to hypothesis '''

                # TODO: Implement an option for using both amr and sdg paths
                if self.include_sdg_path:
                    raise NotImplementedError("Using both SDG and AMR paths is not implemented yet")
                hypothesis = self.get_amr_path(sample)

            elif self.include_sdg_path:
                ''' Add SDG path to hypothesis '''
                hypothesis = self.get_sdg_path(sample)

            sample_inputs = self.parse_one(sample=sample, premise=premise, hypothesis=hypothesis,
                                           max_words_h=self.max_words_h, max_words_p=self.max_words_p,
                                           chars_per_word=self.chars_per_word)
            label = self.label_to_one_hot(label=label)

            sample_result = list(sample_inputs) + [label]
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=self.max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=self.max_words_h, padding='post', truncating='post', value=0.)  # input_word_h
        res[4] = pad_sequences(res[4], maxlen=self.max_words_p, padding='post', truncating='post', value=0.)  # pos_tag_p
        res[5] = pad_sequences(res[5], maxlen=self.max_words_h, padding='post', truncating='post', value=0.)  # pos_tag_h
        res[6] = pad_sequences(res[6], maxlen=self.max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=self.max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h

        # Determine which part of data we need to dump
        if not self.include_exact_match:             del res[6:8]  # Exact match feature
        if not self.include_pos_tags:                del res[4:6]  # Syntactical POS tags
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
        return text, ' '.join(interaction_tuple)

    def get_amr_path(self, sample):
        if sample['amr_path'].strip() != '':
            return sample['amr_path']
        interaction_tuple = sample['interaction_tuple']
        interaction_tuple = [item for item in interaction_tuple if item is not None]
        return ' '.join(interaction_tuple)

    def get_sdg_path(self, sample):
        if sample['sdg_path'].strip() != '':
            return sample['sdg_path']
        interaction_tuple = sample['interaction_tuple']
        interaction_tuple = [item for item in interaction_tuple if item is not None]
        return ' '.join(interaction_tuple)

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
