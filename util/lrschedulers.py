from keras.callbacks import LearningRateScheduler
from keras import backend as K


class CyclicLearningRateScheduler(LearningRateScheduler):
    def __init__(self, lr_min, lr_max, period, verbose=0):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.period = period
        self.delta = (lr_max - lr_min) / (period / 2.)
        self.sign = -1.
        super(CyclicLearningRateScheduler, self).__init__(schedule=self.schedule, verbose=verbose)

    def on_epoch_begin(self, epoch, logs=None):
        super(CyclicLearningRateScheduler, self).on_epoch_begin(epoch, logs)
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def schedule(self, epoch, lr):
        if lr + self.sign * self.delta > self.lr_max:     self.sign *= -1
        if lr + self.sign * self.delta < self.lr_min:     self.sign *= -1

        new_lr = lr + self.sign * self.delta
        assert self.lr_min <= new_lr <= self.lr_max
        return new_lr


class ConstantLearningRateScheduler(LearningRateScheduler):
    def __init__(self, verbose=0):
        super(ConstantLearningRateScheduler, self).__init__(schedule=self.schedule, verbose=verbose)

    def on_epoch_begin(self, epoch, logs=None):
        super(ConstantLearningRateScheduler, self).on_epoch_begin(epoch, logs)
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def schedule(self, epoch, lr):
        return lr
