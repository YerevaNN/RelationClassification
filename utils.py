import numpy as np
import json

np.random.seed(10)


def source_get_data(filename, w2v_model):
    """
        Returns train/val/test sets with regard to labels
    """
    with open(filename, 'r') as f:
        data = json.load(f).items()

    bio_nlp_test = [sample[1] for sample in data if 'bionlp' in sample[0].lower() and
                    'devel' in sample[0].lower()]

    data = [sample[1] for sample in data if not ('bionlp' in sample[0].lower() and
                                                 'devel' in sample[0].lower())]

    train_X, train_Y = process_data_concat(data, w2v_model)
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = split(train_X,
                                                                     train_Y,
                                                                     train=85.,
                                                                     valid=15.,
                                                                     test=0.)

    test_X, test_Y = process_data_concat(bio_nlp_test, w2v_model)
    (test_X, test_Y), (_, _), (_, _) = split(test_X, test_Y, train=100., valid=0., test=0.)

    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)


def get_data(filename):
    with open(filename) as f:
        data = json.load(f).values()

    data = [sample for sample in data if len(sample['interaction_tuple']) == 3 or \
            (len(sample['interaction_tuple']) == 4 and \
             sample['interaction_tuple'][1] == None)]

    for sample in data:
        sample['text'] = sample['text'][1:-1]
        if len(sample['interaction_tuple']) == 4 and sample['interaction_tuple'][1] is None:
            sample['interaction_tuple'] = [sample['interaction_tuple'][0],
                                           sample['interaction_tuple'][2],
                                           sample['interaction_tuple'][3]]

    return data


def normalize(vector):
    return vector / np.linalg.norm(vector)


def word_to_vec(w2v, word):
    if word == None:
        return np.zeros((100,))
    if hasattr(w2v, 'vocab'):  # Means w2v is vanilla w2v model, not a fasttext model
        if word in w2v.vocab:
            return normalize(w2v.word_vec(word))
        current_state = np.random.get_state()
        np.random.seed(hash(word) % 4294967295)
        vector = np.random.normal(0, 0.3, 100)
        np.random.set_state(current_state)
        return normalize(vector)
    else:  # a fasttext case
        return normalize(w2v[word])


def words_to_vec(w2v, words):
    words = words.split()
    return np.mean(np.array([word_to_vec(w2v, word) for word in words]), axis=0)


def process_data_concat(data, w2v_model):
    X, Y = [], []
    for sample in data:
        text = sample['text']
        label = sample['label']
        if len(text) == 0:
            print(sample)
            continue

        interaction_tuple = np.hstack([word_to_vec(w2v_model, word) for word in sample['interaction_tuple']])

        s = np.vstack([np.hstack([interaction_tuple, word_to_vec(w2v_model, word)]) for word in text.split()])
        X.append(s)
        Y.append(label)
    return X, Y


def process_data_attn(data, w2v_model):
    seq, i_tup, y = [], [], []
    for sample in data:
        text = sample['text']
        label = sample['label']
        if len(text) == 0:
            print(sample)
            continue

        interaction_tuple = np.hstack([word_to_vec(w2v_model, word) for word in sample['interaction_tuple']])
        s = np.array([word_to_vec(w2v_model, word) for word in text.split()])
        # print s.shape

        seq.append(s)
        i_tup.append(interaction_tuple)
        y.append(label)
    return (seq, i_tup), y


def process_data_amr(data, w2v_model):
    seq, i_tup, y = [], [], []
    for sample in data:
        text = sample['text']
        label = sample['label']
        if len(text) == 0:
            print(sample)
            continue

        interaction_tuple = np.hstack([word_to_vec(w2v_model, word) for word in sample['interaction_tuple']])

        s = np.array([np.hstack([word_to_vec(w2v_model, word_list[0]), word_to_vec(w2v_model, word_list[1])])
                      if len(word_list) > 1 else
                      np.hstack([word_to_vec(w2v_model, word_list[0]), word_to_vec(w2v_model, None)])
                      for word_list in text])
        # print s.shape

        seq.append(s)
        i_tup.append(interaction_tuple)
        y.append(label)
    return (seq, i_tup), y


def split(samples, targets, train=90., valid=10., test=0.):
    nb_samples = len(samples)
    sum = train + valid + test

    train = int(nb_samples * train // sum)
    valid = int(nb_samples * valid // sum)
    test = int(nb_samples * test // sum)

    indices = np.random.permutation(nb_samples)

    train_X = samples[:train]
    train_Y = targets[:train]
    valid_X = samples[train:nb_samples - test]
    valid_Y = targets[train:nb_samples - test]
    test_X = samples[nb_samples - test:]
    test_Y = targets[nb_samples - test:]

    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)
