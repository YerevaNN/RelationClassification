import random


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
