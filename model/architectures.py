from keras import Model, Input
from keras import backend as K
from keras.layers import Dense, Concatenate, GRU, Masking, Conv2D
from keras.layers.wrappers import Bidirectional
from keras.models import load_model

from feature_extractors.densenet import DenseNet
from layers.decaying_dropout import DecayingDropout
from layers.encoding import Encoding
from layers.interaction import Interaction
from model.input import WordVectorInput, CharInput, PosTagInput, ExactMatchInput
from optimizers.l2optimizer import L2Optimizer


class Classifier(Model):
    def __init__(self, input_shapes=None,
                 include_word_vectors=True, word_embedding_weights=None, train_word_embeddings=True,
                 include_chars=True, chars_per_word=16, char_embedding_size=8,
                 include_pos_tag_features=True, nb_pos_tags=20, pos_tag_embedding_size=8,
                 include_exact_match=True,
                 nb_labels=3,
                 inputs=None, outputs=None, name='RelationClassifier'):
        if inputs or outputs:
            super(Classifier, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        self.input_shapes = input_shapes
        self.include_word_vectors = include_word_vectors
        self.word_embedding_weights = word_embedding_weights
        self.train_word_embeddings = train_word_embeddings
        self.include_chars = include_chars
        self.chars_per_word = chars_per_word
        self.char_embedding_size = char_embedding_size
        self.include_pos_tag_features = include_pos_tag_features
        self.pos_tag_embedding_size = pos_tag_embedding_size
        self.nb_pos_tags = nb_pos_tags
        self.include_exact_match = include_exact_match

        inputs, embeddings = self.create_inputs()
        embeddings = [Concatenate()(embedding) for embedding in embeddings]

        encodings = self.create_encodings(embeddings)
        interaction = self.create_interaction(encodings)
        features = self.create_feature_extraction(interaction)
        out = Dense(units=nb_labels, activation='softmax', name='Output')(features)
        super(Classifier, self).__init__(inputs=inputs, outputs=out, name=name)

    def create_inputs(self):
        creators = [f for (f, include) in zip([WordVectorInput(shapes=self.input_shapes,
                                                               word_embedding_weights=self.word_embedding_weights,
                                                               train_word_embeddings=self.train_word_embeddings),
                                               CharInput(shapes=self.input_shapes,
                                                         chars_per_word=self.chars_per_word,
                                                         embedding_size=self.char_embedding_size),
                                               PosTagInput(shapes=self.input_shapes,
                                                           nb_pos_tags=self.nb_pos_tags,
                                                           embedding_size=self.pos_tag_embedding_size),
                                               ExactMatchInput(shapes=self.input_shapes)],
                                              [self.include_word_vectors,
                                               self.include_chars,
                                               self.include_pos_tag_features,
                                               self.include_exact_match]) if include]
        inputs = []
        embeddings = [[], []]
        for create_input in creators:
            net_inputs, net_embeddings = create_input()
            inputs += net_inputs
            embeddings[0].append(net_embeddings[0])
            embeddings[1].append(net_embeddings[1])

        return inputs, embeddings

    def create_encodings(self, embeddings):
        raise NotImplementedError('You need to create encoding for each embedding')

    def create_interaction(self, encodings):
        raise NotImplementedError('You need to create interaction of encodings')

    def create_feature_extraction(self, interaction):
        raise NotImplementedError('You need to create feature extraction of features')


class BiGRUClassifier(Classifier):
    def __init__(self, dropout_rate=0.3, **kwargs):
        self.dropout_rate = dropout_rate
        super(BiGRUClassifier, self).__init__(**kwargs)

    def create_encodings(self, embeddings):
        embeddings = [Masking()(embedding) for embedding in embeddings]
        return [Bidirectional(GRU(units=64,
                                  return_sequences=True,
                                  dropout=self.dropout_rate))(embedding) for embedding in embeddings]

    def create_interaction(self, encodings):
        concat = Concatenate(axis=1)(encodings)
        interaction = Bidirectional(GRU(units=128, dropout=self.dropout_rate))(concat)
        return interaction

    def create_feature_extraction(self, interaction):
        return interaction


class DIIN(Classifier):
    def __init__(self, first_scale_down_ratio=0.3, transition_scale_down_ratio=0.5, growth_rate=20,
                 layers_per_dense_block=8, nb_dense_blocks=3,
                 dropout_initial_keep_rate=1., dropout_decay_rate=0.977, dropout_decay_interval=10000,
                 **kwargs):
        self.first_scale_down_ratio = first_scale_down_ratio
        self.transition_scale_down_ratio = transition_scale_down_ratio
        self.growth_rate = growth_rate
        self.layers_per_dense_block = layers_per_dense_block
        self.nb_dense_blocks = nb_dense_blocks
        self.dropout_initial_keep_rate = dropout_initial_keep_rate
        self.dropout_decay_rate = dropout_decay_rate
        self.dropout_decay_interval = dropout_decay_interval
        super(DIIN, self).__init__(**kwargs)

    def create_encodings(self, embeddings):
        encodings = [Encoding()(embedding) for embedding in embeddings]
        return [DecayingDropout(initial_keep_rate=self.dropout_initial_keep_rate,
                                decay_interval=self.dropout_decay_interval,
                                decay_rate=self.dropout_decay_rate)(encoding) for encoding in encodings]

    def create_interaction(self, encodings):
        interaction = Interaction(name='Interaction')(encodings)
        return DecayingDropout(initial_keep_rate=self.dropout_initial_keep_rate,
                               decay_interval=self.dropout_decay_interval,
                               decay_rate=self.dropout_decay_rate)(interaction)

    def create_feature_extraction(self, interaction):
        d = K.int_shape(interaction)[-1]
        feature_extractor_input = Conv2D(filters=int(d * self.first_scale_down_ratio),
                                         kernel_size=1,
                                         activation=None,
                                         name='FirstScaleDown')(interaction)
        feature_extractor = DenseNet(include_top=False,
                                     input_tensor=Input(shape=K.int_shape(feature_extractor_input)[1:]),
                                     nb_dense_block=self.nb_dense_blocks,
                                     nb_layers_per_block=self.layers_per_dense_block,
                                     compression=self.transition_scale_down_ratio,
                                     growth_rate=self.growth_rate)(feature_extractor_input)

        features = DecayingDropout(initial_keep_rate=self.dropout_initial_keep_rate,
                                   decay_interval=self.dropout_decay_interval,
                                   decay_rate=self.dropout_decay_rate,
                                   name='Features')(feature_extractor)
        return features


def get_classifier(model_path=None,
                   architecture='BiGRU',
                   input_shapes=None,
                   include_word_vectors=True, word_embedding_weights=None, train_word_embeddings=True,
                   include_chars=True, chars_per_word=16, char_embedding_size=8,
                   include_pos_tag_features=True, nb_pos_tags=50, pos_tag_embedding_size=8,
                   include_exact_match=True,
                   nb_labels=3,
                   first_scale_down_ratio=0.3, transition_scale_down_ratio=0.5, growth_rate=20,
                   layers_per_dense_block=8, nb_dense_blocks=3,
                   dropout_initial_keep_rate=1., dropout_decay_rate=0.977, dropout_decay_interval=10000):
    if model_path:
        return load_model(model_path, custom_objects={'Classifier': Classifier,
                                                      'BiGRUClassifier': BiGRUClassifier,
                                                      'DIIN': DIIN,
                                                      'Encoding': Encoding,
                                                      'Interaction': Interaction,
                                                      'DenseNet': DenseNet,
                                                      'DecayingDropout': DecayingDropout,
                                                      'L2Optimizer': L2Optimizer})
    if architecture == 'BiGRU':
        return BiGRUClassifier(input_shapes=input_shapes,
                               include_word_vectors=include_word_vectors,
                               word_embedding_weights=word_embedding_weights,
                               train_word_embeddings=train_word_embeddings,
                               include_chars=include_chars,
                               chars_per_word=chars_per_word, char_embedding_size=char_embedding_size,
                               include_pos_tag_features=include_pos_tag_features,
                               nb_pos_tags=nb_pos_tags, pos_tag_embedding_size=pos_tag_embedding_size,
                               include_exact_match=include_exact_match,
                               dropout_rate=1.-dropout_initial_keep_rate,
                               nb_labels=nb_labels)
    elif architecture == 'DIIN':
        return DIIN(input_shapes=input_shapes,
                    include_word_vectors=include_word_vectors,
                    word_embedding_weights=word_embedding_weights,
                    train_word_embeddings=train_word_embeddings,
                    include_chars=include_chars,
                    chars_per_word=chars_per_word, char_embedding_size=char_embedding_size,
                    include_pos_tag_features=include_pos_tag_features,
                    nb_pos_tags=nb_pos_tags, pos_tag_embedding_size=pos_tag_embedding_size,
                    include_exact_match=include_exact_match,
                    nb_labels=nb_labels,
                    first_scale_down_ratio=first_scale_down_ratio,
                    transition_scale_down_ratio=transition_scale_down_ratio,
                    growth_rate=growth_rate,
                    layers_per_dense_block=layers_per_dense_block,
                    nb_dense_blocks=nb_dense_blocks,
                    dropout_initial_keep_rate=dropout_initial_keep_rate,
                    dropout_decay_rate=dropout_decay_rate,
                    dropout_decay_interval=dropout_decay_interval)
    else:
        raise NotImplementedError('No implementation found for the specified architecture')
