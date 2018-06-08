from keras import Input, Sequential
from keras.engine import InputLayer
from keras.layers import Embedding, TimeDistributed, Bidirectional, GRU


class InputCreator(object):
    """ InputCreator is the contract that needs to be implemented by any class that implements how a specific input
        of a network is created """
    def __init__(self, shapes):
        self.shapes = shapes

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('You need to implement this method')


class WordVectorInput(InputCreator):

    def __init__(self, shapes, word_embedding_weights, train_word_embeddings=True):
        if word_embedding_weights is None:
            raise ValueError('You need to provide valid word embedding weights')
        self.word_embedding = Embedding(input_dim=word_embedding_weights.shape[0],
                                        output_dim=word_embedding_weights.shape[1],
                                        weights=[word_embedding_weights],
                                        trainable=train_word_embeddings,
                                        name='WordEmbedding')
        super(WordVectorInput, self).__init__(shapes=shapes)

    def __call__(self, *args, **kwargs):
        inputs = [Input(shape=(shape,), dtype='int64') for shape in self.shapes]
        embeddings = [self.word_embedding(word_input) for word_input in inputs]
        return inputs, embeddings


class CharInput(InputCreator):
    def __init__(self, shapes, chars_per_word, embedding_size, input_dim=100):
        self.chars_per_word = chars_per_word
        self.embedding_size = embedding_size
        self.input_dim = input_dim
        super(CharInput, self).__init__(shapes=shapes)

    def __call__(self, *args, **kwargs):
        # Share weights of character-level embedding
        embedding_layer = TimeDistributed(Sequential([
            InputLayer(input_shape=(self.chars_per_word,)),
            Embedding(input_dim=self.input_dim,
                      output_dim=self.embedding_size,
                      input_length=self.chars_per_word,
                      mask_zero=True),
            Bidirectional(GRU(units=24))
        ]), name='CharEmbedding')

        inputs = [Input(shape=(shape, self.chars_per_word)) for shape in self.shapes]
        embeddings = [embedding_layer(char_input) for char_input in inputs]
        return inputs, embeddings


class PosTagInput(InputCreator):
    def __init__(self, shapes, nb_pos_tags, embedding_size):
        self.pos_tag_embedding = Embedding(input_dim=nb_pos_tags,
                                           output_dim=embedding_size,
                                           trainable=True,
                                           name='PosTagEmbedding')
        super(PosTagInput, self).__init__(shapes=shapes)

    def __call__(self, *args, **kwargs):
        inputs = [Input(shape=(shape,), dtype='int64') for shape in self.shapes]
        embeddings = [self.pos_tag_embedding(pos_tag_input) for pos_tag_input in inputs]
        return inputs, embeddings
