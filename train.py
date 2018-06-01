from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import random
from pprint import pprint

import fire
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from sklearn.utils import class_weight

from data.mappings import WordVectors, CharToIdMapping, KeyToIdMapping
from data.preprocess import BioNLPPreprocessor
from model.architectures import get_classifier
from util.generators import data_generator
from util.lrschedulers import CyclicLearningRateScheduler, ConstantLearningRateScheduler
from util.util import get_word2vec_file_path, AllMetrics, get_git_hash

try:                import cPickle as pickle
except ImportError: import _pickle as pickle


def train(batch_size=80, p=60, h=22, epochs=70, steps_per_epoch=500, patience=5,
          chars_per_word=20, char_embed_size=8,
          pos_tag_embedding_size=8,
          l2_full_step=100000, l2_full_ratio=9e-5, l2_difference_penalty=1e-3,
          first_scale_down_ratio=0.3, transition_scale_down_ratio=0.5, growth_rate=20,
          layers_per_dense_block=8, nb_dense_blocks=3,
          dropout_initial_keep_rate=1., dropout_decay_rate=0.977, dropout_decay_interval=10000,
          lr_schedule='constant', lr=0.001,
          lr_max=1., lr_min=0.1, lr_period=3,
          random_seed=777,
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

    ''' Fix random seed for reproducibility '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:                    import tensorflow; tensorflow.set_random_seed(random_seed)
    except ImportError:     pass
    
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

    ''' Prepare the model '''
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

    ''' Prepare optimizers and learning rate schedulers '''
    if lr_schedule == 'constant':   optimizer, lr_scheduler = Adam(lr=lr), ConstantLearningRateScheduler()
    elif lr_schedule == 'cyclic':   optimizer, lr_scheduler = SGD(lr=lr_max), CyclicLearningRateScheduler(lr_min, lr_max, period=lr_period)
    else:                           raise NotImplementedError('Cannot find implementation for the specified schedule')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    ''' Initialize training '''
    print('Loading data...')
    train_samples = train_processor.load_data(train_path)
    valid_samples = valid_processor.load_data(valid_path)
    train_samples = [sample for sample in train_samples if not train_processor.skip_sample(sample)]
    valid_samples = [sample for sample in valid_samples if not valid_processor.skip_sample(sample)]
    valid_data = valid_processor.parse(valid_samples, verbose=True)

    ''' Give weights to classes '''
    y_train = np.array([train_processor.get_label(sample) for sample in train_samples])
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print('Class weights: ', class_weights)

    model.fit_generator(generator=data_generator(samples=train_samples, processor=train_processor, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs,
                        callbacks=[AllMetrics(valid_data[:-1], valid_data[-1]),
                                   lr_scheduler,
                                   TensorBoard(log_dir=log_dir),
                                   ModelCheckpoint(filepath=os.path.join(models_dir, 'model-{epoch:02d}-f1-{val_f1:.2f}.hdf5'), monitor='val_f1', save_best_only=True, verbose=1, mode='max'),
                                   EarlyStopping(patience=patience)],
                        class_weight=class_weights)


if __name__ == '__main__':
    fire.Fire(train)
