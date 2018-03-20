from __future__ import print_function

import json

import fire
import numpy as np
from keras.models import load_model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

from preprocess import BioNLPPreprocessor
from layers.decaying_dropout import DecayingDropout
from model import Classifier
from optimizers.l2optimizer import L2Optimizer
from util import precision, recall, f1
try:                import cPickle as pickle
except ImportError: import _pickle as pickle


def predict(model, preprocessor, data, output_path, batch_size=70):

    eval_predictions = []
    eval_labels = [preprocessor.get_label(sample) for sample in data]
    for batch_start in tqdm(range(0, len(data), batch_size)):
        batch = data[batch_start: batch_start + batch_size]
        data_input = preprocessor.parse(data=batch)
        data_input = data_input[:-1]    # Last axis contains real labels

        model_outputs = model.predict(data_input)
        predictions = np.argmax(model_outputs, axis=1)

        eval_predictions += list(predictions.flatten())
        for sample, prediction in zip(batch, predictions):
            sample['prediction'] = int(prediction)

    print('Confusion Matrix:\n', confusion_matrix(eval_labels, eval_predictions))
    print('F score:', f1_score(eval_labels, eval_predictions))
    print('Accuracy:', accuracy_score(eval_labels, eval_predictions))

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=True)


def main(model_path, batch_size=80, dataset='bionlp', processor_path='data/processor.pkl',
         input_path='data/bionlp_test_data.json', output_path='data/out_bionlp_test_data.json'):

    if dataset == 'bionlp':
        model = load_model(model_path, custom_objects={'Classifier': Classifier,
                                                       'DecayingDropout': DecayingDropout,
                                                       'L2Optimizer': L2Optimizer,
                                                       'precision': precision,
                                                       'recall': recall,
                                                       'f1': f1})
        with open(input_path, 'r') as f:        data = json.load(f)
        with open(processor_path, 'rb') as f:   preprocessor = pickle.load(f)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')

    predict(model=model,
            preprocessor=preprocessor,
            data=data,
            output_path=output_path,
            batch_size=batch_size)


if __name__ == '__main__':
    fire.Fire(main)
