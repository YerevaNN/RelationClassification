from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import io
import random
import fire

from pprint import pprint

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from data.mappings import WordVectors, CharToIdMapping, KeyToIdMapping
from model.architectures import get_classifier
from optimizers.l2optimizer import L2Optimizer
from data.preprocess import BioNLPPreprocessor
from util import get_word2vec_file_path, AllMetrics, get_git_hash

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


def train(batch_size=80, p=60, h=22, epochs=70, steps_per_epoch=500,
          chars_per_word=20, char_embed_size=8,
          pos_tag_embedding_size=8,
          l2_full_step=100000, l2_full_ratio=9e-5, l2_difference_penalty=1e-3,
          first_scale_down_ratio=0.3, transition_scale_down_ratio=0.5, growth_rate=20,
          layers_per_dense_block=8, nb_dense_blocks=3,
          dropout_initial_keep_rate=1., dropout_decay_rate=0.977, dropout_decay_interval=10000,
          architecture='BiGRU',
          models_dir='models', log_dir='logs',
          train_path='data/bionlp_train_data.json',
          valid_path='data/bionlp_valid_data.json',
          train_processor_load_path=None, train_processor_save_path='data/train_processor.pkl',
          valid_processor_load_path=None, valid_processor_save_path='data/valid_processor.pkl',
          word_vec_load_path=None, max_word_vecs=None, normalize_word_vectors=False, train_word_embeddings=False,
          dataset='bionlp',
          train_interaction=None, valid_interaction=None,
          omit_single_interaction=False, omit_amr_path=False, omit_sdg_path=False,
          omit_word_vectors=False, omit_chars=False,
          omit_pos_tags=False, omit_exact_match=False):

    # Create directories if they are not present
    if not os.path.exists(models_dir):  os.mkdir(models_dir)
    if not os.path.exists(log_dir):     os.mkdir(log_dir)
    logs = locals()
    logs['commit'] = get_git_hash()
    with io.open(os.path.join(log_dir, 'info.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(logs, ensure_ascii=False, indent=True))
    pprint(logs)
    
    ''' Prepare data '''
    if dataset == 'bionlp':
        train_processor = BioNLPPreprocessor(max_words_p=p, max_words_h=h, chars_per_word=chars_per_word,
                                             include_word_vectors=not omit_word_vectors,
                                             include_chars=not omit_chars,
                                             include_pos_tags=not omit_pos_tags,
                                             include_exact_match=not omit_exact_match,
                                             include_amr_path=not omit_amr_path,
                                             include_sdg_path=not omit_sdg_path,
                                             include_single_interaction=not omit_single_interaction)
        valid_processor = BioNLPPreprocessor(max_words_p=p, max_words_h=h, chars_per_word=chars_per_word,
                                             include_word_vectors=not omit_word_vectors,
                                             include_chars=not omit_chars,
                                             include_pos_tags=not omit_pos_tags,
                                             include_exact_match=not omit_exact_match,
                                             include_amr_path=not omit_amr_path,
                                             include_sdg_path=not omit_sdg_path,
                                             include_single_interaction=not omit_single_interaction)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')

    if train_processor_load_path is None and valid_processor_load_path is None:
        all_words, all_parts_of_speech = train_processor.get_all_words_with_parts_of_speech([train_path, valid_path])

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
        if train_interaction is not None:  train_processor.valid_interactions = {train_interaction}
        if valid_interaction is not None:  valid_processor.valid_interactions = {valid_interaction}

    elif train_processor_load_path is not None and valid_processor_load_path is not None:
        with open(train_processor_load_path, 'rb') as f:    train_processor = pickle.load(file=f)
        with open(valid_processor_load_path, 'rb') as f:    valid_processor = pickle.load(file=f)
    else:
        raise ValueError('Both --train_processor_load_path and --valid_processor_load_path need to be provided, '
                         'or both omitted')

    with open(train_processor_save_path, 'wb') as f:    pickle.dump(train_processor, file=f, protocol=2)
    with open(valid_processor_save_path, 'wb') as f:    pickle.dump(valid_processor, file=f, protocol=2)

    ''' Prepare the model and optimizers '''
    model = get_classifier(architecture=architecture,
                           input_shapes=(None, None),  # of (p, h)
                           include_word_vectors=not omit_word_vectors,
                           word_embedding_weights=train_processor.word_mapping.vectors,
                           train_word_embeddings=train_word_embeddings,
                           include_chars=not omit_chars,
                           chars_per_word=chars_per_word,
                           char_embedding_size=char_embed_size,
                           include_pos_tag_features=not omit_pos_tags,
                           nb_pos_tags=len(train_processor.part_of_speech_mapping.key_to_id),
                           pos_tag_embedding_size=pos_tag_embedding_size,
                           include_exact_match=not omit_exact_match,
                           nb_labels=len(train_processor.get_labels()),
                           first_scale_down_ratio=first_scale_down_ratio,
                           transition_scale_down_ratio=transition_scale_down_ratio,
                           growth_rate=growth_rate,
                           layers_per_dense_block=layers_per_dense_block,
                           nb_dense_blocks=nb_dense_blocks,
                           dropout_initial_keep_rate=dropout_initial_keep_rate,
                           dropout_decay_rate=dropout_decay_rate,
                           dropout_decay_interval=dropout_decay_interval)
    adam = L2Optimizer(Adam(3e-4), l2_full_step, l2_full_ratio, l2_difference_penalty)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    ''' Initialize training '''
    print('Loading data...')
    train_samples = train_processor.load_data(train_path)
    valid_samples = valid_processor.load_data(valid_path)
    train_samples = [sample for sample in train_samples if not train_processor.skip_sample(sample)]
    valid_samples = [sample for sample in valid_samples if not valid_processor.skip_sample(sample)]
    valid_data = valid_processor.parse(valid_samples, verbose=True)

    ''' Give weights to classes '''
    zer = 1. * sum([train_processor.get_label(sample) == 0 for sample in train_samples])
    one = 1. * sum([train_processor.get_label(sample) == 1 for sample in train_samples])
    class_weights = [len(train_samples) / zer, len(train_samples) / one]
    print('Class weights: ', class_weights)

    model.fit_generator(generator=data_generator(samples=train_samples, processor=train_processor, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=(valid_data[:-1], valid_data[-1]),
                        callbacks=[TensorBoard(log_dir=log_dir),
                                   ModelCheckpoint(filepath=os.path.join(models_dir, 'model.{epoch:02d}-{val_loss:.2f}.hdf5')),
                                   EarlyStopping(patience=20),
                                   AllMetrics(valid_data[:-1], valid_data[-1])],
                        class_weight=class_weights)


if __name__ == '__main__':
    fire.Fire(train)
