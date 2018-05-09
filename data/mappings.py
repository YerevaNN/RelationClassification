import io

import numpy as np
from tqdm import tqdm


class Mapping(object):
    def __init__(self):
        self.UNK = '<<UNK>>'

    def __getitem__(self, item):
        raise NotImplementedError('You need to implement the mapping operator...')


class KeyToIdMapping(Mapping):
    def __init__(self, keys, include_unknown=True):
        super(KeyToIdMapping, self).__init__()
        if include_unknown: self.keys = [self.UNK]
        else:               self.keys = []
        self.keys += list(keys)
        self.key_to_id = {key: i for i, key in enumerate(self.keys)}

    def __getitem__(self, item):
        if item in self.key_to_id:      return self.key_to_id[item]
        if self.UNK in self.key_to_id:  return self.key_to_id[self.UNK]
        raise KeyError('`{}` is not present in the mapping'.format(item))

    def __len__(self):
        return len(self.keys)


class CharToIdMapping(KeyToIdMapping):
    def __init__(self, words, include_unknown=True):
        chars = set()
        for word in set(words):
            chars = chars.union(set(word))
        super(CharToIdMapping, self).__init__(chars, include_unknown)


class WordVectors(Mapping):
    def __init__(self):
        super(WordVectors, self).__init__()
        self.words = []
        self.vectors = []
        self.word_to_id = {}
        self.vector_size = None

    def get_not_present_word_vectors(self, not_present_words, normalize):
        words = []
        vectors = []
        for word in not_present_words:
            vec = np.random.uniform(size=self.vector_size)
            if normalize:
                vec /= np.linalg.norm(vec, ord=2)
            words.append(word)
            vectors.append(vec)
        return words, vectors

    def load_fast_text(self, file_path, needed_words=None, include_unknown=True):
        import fastText
        assert file_path.endswith('.bin')
        embeddings = fastText.load_model(file_path)

        if needed_words is None:
            needed_words = set()
        if include_unknown:
            needed_words.add(self.UNK)

        self.words = list(needed_words)
        self.vectors = [embeddings.get_word_vector(word) for word in self.words]
        self.vectors = np.array(self.vectors)

        print('Initializing word mappings...')
        self.vector_size = self.vectors.shape[-1]
        self.word_to_id  = {word: i for i, word in enumerate(self.words)}
        self.vectors = np.array(self.vectors, copy=False)

        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def load(self, file_path, separator=' ', normalize=True, max_words=None, needed_words=None, include_unknown=True):
        """
        :return: words[], np.array(vectors)
        """
        if file_path.endswith('.bin'):
            self.load_fast_text(file_path=file_path, needed_words=needed_words, include_unknown=include_unknown)
            return

        seen_words = set()
        self.words = []
        self.vectors = []

        print('Loading', file_path)
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.replace(' \n', '').split(separator)
                word = values[0]

                # Skip unnecessary words
                if needed_words is not None and word not in needed_words:
                    continue

                if len(values) < 5 or word in seen_words:
                    print('Invalid word:', word)
                    continue

                seen_words.add(word)
                vec = np.asarray(values[1:], dtype='float32')
                if normalize:
                    vec /= np.linalg.norm(vec, ord=2)

                if self.vector_size is None:
                    self.vector_size = len(vec)
                elif len(vec) != self.vector_size:
                    print('Vector size not consistent: skipping', word)
                    continue

                self.words.append(word)
                self.vectors.append(vec)
                if max_words and len(self.words) >= max_words:
                    break

        if needed_words is None:
            needed_words = set()
        if include_unknown:
            needed_words.add(self.UNK)

        present_words = set(self.words)
        not_present_words = needed_words - present_words
        print('#Present words:', len(present_words), '\t#Not present words', len(not_present_words))

        not_present_words, not_present_vectors = self.get_not_present_word_vectors(not_present_words=not_present_words,
                                                                                   normalize=normalize)
        self.words += not_present_words
        self.vectors += not_present_vectors

        print('Initializing word mappings...')
        self.word_to_id  = {word: i for i, word in enumerate(self.words)}
        self.vectors = np.array(self.vectors, copy=False)

        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def get_word(self, index):
        return self.words[index]

    def get_vector(self, word):
        if type(word) is int:               return self.vectors[word]
        if type(word) is str:               return self.vectors[self[word]]
        raise ValueError('Key has to be either int or str')

    def __getitem__(self, word):
        if word in self.word_to_id:         return self.word_to_id[word]
        if self.UNK in self.word_to_id:     return self.word_to_id[self.UNK]
        raise KeyError('`{}` word is not present in the vectors, you can provide include_unknown=True in '
                       'load_word_vectors() to return the word vector corresponding to <UNK> in these cases.'
                       .format(word))

    def __len__(self):
        return len(self.words)
