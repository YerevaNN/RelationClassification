from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import fire

from pprint import pprint

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from data.mappings import WordVectors, CharToIdMapping, KeyToIdMapping
from model import Classifier
from optimizers.l2optimizer import L2Optimizer
from preprocess import BioNLPPreprocessor
from util import get_word2vec_file_path, precision, recall, f1
try:                import cPickle as pickle
except ImportError: import _pickle as pickle


def data_generator(samples, processor, batch_size, shuffle=True):
    batch_start = len(samples)
    indices = list(range(len(samples)))
    while True:
        ''' Start a new epoch '''
        if batch_start >= len(samples):
            batch_start = 0
            if shuffle:
                random.shuffle(indices)

        ''' Generate a new batch '''
        batch = [samples[i] for i in indices[batch_start: batch_start + batch_size]]
        batch_start += batch_size
        batch = processor.parse(data=batch)
        inputs, labels = batch[:-1], batch[-1]
        yield inputs, labels


def train(batch_size=80, p=22, h=4, epochs=70, steps_per_epoch=500, valid_omit_interaction=None,
          chars_per_word=18, char_embed_size=8,
          dropout_initial_keep_rate=1., dropout_decay_rate=0.977, dropout_decay_interval=10000,
          l2_full_step=100000, l2_full_ratio=9e-5, l2_difference_penalty=1e-3,
          load_dir='data', models_dir='models', log_dir='logs',
          processor_path='data/processor.pkl',
          word_vec_load_path=None, max_word_vecs=None, normalize_word_vectors=False, train_word_embeddings=True,
          dataset='bionlp',
          omit_word_vectors=False, omit_chars=False, omit_syntactical_features=False, omit_exact_match=False):
    """
    Train the model
    """
    pprint(locals())
    
    ''' Prepare data '''
    if dataset == 'bionlp':
        train_processor = BioNLPPreprocessor(max_words_p=p,
                                             max_words_h=h,
                                             chars_per_word=chars_per_word,
                                             include_word_vectors=not omit_word_vectors,
                                             include_chars=not omit_chars,
                                             include_syntactical_features=not omit_syntactical_features,
                                             include_exact_match=not omit_exact_match)
        valid_processor = BioNLPPreprocessor(max_words_p=p,
                                             max_words_h=h,
                                             chars_per_word=chars_per_word,
                                             include_word_vectors=not omit_word_vectors,
                                             include_chars=not omit_chars,
                                             include_syntactical_features=not omit_syntactical_features,
                                             include_exact_match=not omit_exact_match)

        train_path = os.path.join(load_dir, 'bionlp_train_data.json')
        test_path = os.path.join(load_dir, 'bionlp_test_data.json')
        dev_path = os.path.join(load_dir, 'bionlp_valid_data.json')
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')

    all_words, all_parts_of_speech = train_processor.get_all_words_with_parts_of_speech([train_path, test_path, dev_path])

    ''' Mappings '''
    word_mapping = WordVectors()
    word_mapping.load(file_path=get_word2vec_file_path(word_vec_load_path),
                      needed_words=set(all_words),
                      normalize=normalize_word_vectors,
                      max_words=max_word_vecs)

    char_mapping = CharToIdMapping(set(all_words))
    part_of_speech_mapping = KeyToIdMapping(keys=set(all_parts_of_speech))

    ''' Initialize preprocessor mappings '''
    valid_processor.word_mapping = train_processor.word_mapping = word_mapping
    valid_processor.char_mapping = train_processor.char_mapping = char_mapping
    valid_processor.part_of_speech_mapping = train_processor.part_of_speech_mapping = part_of_speech_mapping
    if valid_omit_interaction is not None:
        valid_processor.omit_interactions = {valid_omit_interaction}

    if processor_path is not None:
        print('Saving processor...')
        with open(processor_path, 'wb') as f:
            pickle.dump(train_processor, file=f)

    ''' Prepare the model and optimizers '''
    model = Classifier(p=None,  # or p
                       h=None,  # or h
                       include_word_vectors=not omit_word_vectors,
                       word_embedding_weights=train_processor.word_mapping.vectors,
                       train_word_embeddings=train_word_embeddings,
                       include_chars=not omit_chars,
                       chars_per_word=chars_per_word,
                       char_embedding_size=char_embed_size,
                       include_syntactical_features=not omit_syntactical_features,
                       syntactical_feature_size=len(train_processor.part_of_speech_mapping.key_to_id),
                       include_exact_match=not omit_exact_match,
                       dropout_initial_keep_rate=dropout_initial_keep_rate,
                       dropout_decay_rate=dropout_decay_rate,
                       dropout_decay_interval=dropout_decay_interval,
                       nb_labels=len(train_processor.get_labels()))
    adam = L2Optimizer(Adam(3e-4), l2_full_step, l2_full_ratio, l2_difference_penalty)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1])

    ''' Initialize training '''
    print('Loading data...')
    train_samples = train_processor.load_data(train_path)
    valid_samples = valid_processor.load_data(dev_path)

    ''' Give weights to classes '''
    one = sum([train_processor.get_label(sample) for sample in train_samples])
    two = len(train_samples) - one
    class_weights = [len(train_samples) / (1. * one), len(train_samples) / (1. * two)]

    model.fit_generator(generator=data_generator(samples=train_samples, processor=train_processor, batch_size=batch_size),
                        validation_data=data_generator(samples=valid_samples, processor=valid_processor, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=len(valid_samples) // batch_size,
                        epochs=epochs,
                        callbacks=[TensorBoard(log_dir=log_dir),
                                   ModelCheckpoint(filepath=os.path.join(models_dir, 'model.{epoch:02d}-{val_loss:.2f}.hdf5')),
                                   EarlyStopping(patience=3)],
                        class_weight=class_weights)


if __name__ == '__main__':
    fire.Fire(train)
