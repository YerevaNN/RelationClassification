from keras import Model
from keras.layers import Dense, Concatenate, GRU, Masking
from keras.layers.wrappers import Bidirectional

from model.input import WordVectorInput, CharInput, PosTagInput, ExactMatchInput


class Classifier(Model):
    def __init__(self, nb_labels=3,
                 inputs=None, outputs=None, name='RelationClassifier'):
        if inputs or outputs:
            super(Classifier, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        inputs, embeddings = self.create_inputs()
        embeddings = [Concatenate()(embedding) for embedding in embeddings]
        embeddings = [Masking()(embedding) for embedding in embeddings]

        encodings = self.create_encodings(embeddings)
        interaction = self.create_interaction(encodings)
        features = self.create_feature_extraction(interaction)
        out = Dense(units=nb_labels, activation='softmax', name='Output')(features)
        super(Classifier, self).__init__(inputs=inputs, outputs=out, name=name)

    def create_inputs(self):
        raise NotImplementedError('You need to implement input creation')

    def create_encodings(self, embeddings):
        raise NotImplementedError('You need to create encoding for each embedding')

    def create_interaction(self, encodings):
        raise NotImplementedError('You need to create interaction of encodings')

    def create_feature_extraction(self, interaction):
        raise NotImplementedError('You need to create feature extraction of features')


class BiGRUClassifier(Classifier):
    def __init__(self, input_shapes=None,
                 include_word_vectors=True, word_embedding_weights=None, train_word_embeddings=True,
                 include_chars=True, chars_per_word=16, char_embedding_size=8,
                 include_postag_features=True, postag_feature_size=50,
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
        self.include_postag_features = include_postag_features
        self.postag_feature_size = postag_feature_size
        self.include_exact_match = include_exact_match

        super(BiGRUClassifier, self).__init__(nb_labels=nb_labels,
                                              inputs=inputs, outputs=outputs, name=name)

    def create_inputs(self):
        creators = [f for (f, include) in zip([WordVectorInput(shapes=self.input_shapes,
                                                               word_embedding_weights=self.word_embedding_weights,
                                                               train_word_embeddings=self.train_word_embeddings),
                                               CharInput(shapes=self.input_shapes,
                                                         chars_per_word=self.chars_per_word,
                                                         embedding_size=self.char_embedding_size),
                                               PosTagInput(shapes=self.input_shapes,
                                                           postag_feature_size=self.postag_feature_size),
                                               ExactMatchInput(shapes=self.input_shapes)],
                                              [self.include_word_vectors,
                                               self.include_chars,
                                               self.include_postag_features,
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
        return [Bidirectional(GRU(units=64, return_sequences=True))(embedding) for embedding in embeddings]

    def create_interaction(self, encodings):
        concat = Concatenate(axis=1)(encodings)
        interaction = Bidirectional(GRU(units=128))(concat)
        return interaction

    def create_feature_extraction(self, interaction):
        return interaction


def get_classifier(architecture='BiGRU',
                   input_shapes=None,
                   include_word_vectors=True, word_embedding_weights=None, train_word_embeddings=True,
                   include_chars=True, chars_per_word=16, char_embedding_size=8,
                   include_postag_features=True, postag_feature_size=50,
                   include_exact_match=True,
                   nb_labels=3):
    if architecture == 'BiGRU':
        return BiGRUClassifier(input_shapes=input_shapes,
                               include_word_vectors=include_word_vectors,
                               word_embedding_weights=word_embedding_weights,
                               train_word_embeddings=train_word_embeddings,
                               include_chars=include_chars,
                               chars_per_word=chars_per_word, char_embedding_size=char_embedding_size,
                               include_postag_features=include_postag_features, postag_feature_size=postag_feature_size,
                               include_exact_match=include_exact_match,
                               nb_labels=nb_labels)
    else:
        raise NotImplementedError('No implementation found for the specified architecture')
