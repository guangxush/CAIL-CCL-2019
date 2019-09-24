# -*- coding: utf-8 -*-
from data_preprocess import load_data
from config import Config
from models import *
import os, codecs

train = True  # true表示训练 False表示测试
level = 'word'
fasttext = False
overwrite = False
swa = False
model_name = 'esim'


def get_data(train_file=None, test_file=None, level='word'):
    if level == 'word':
        x_a_train, x_b_train, x_c_train, vocab = load_data(train_file, level=level)
        x_a_test, x_b_test, _ = load_data(test_file, level=level, test=True)
        return [x_a_train, x_b_train, x_c_train], [x_a_test, x_b_test], vocab
    else:
        x_a_train, x_b_train, x_c_train, x_a_char_train, x_b_char_train, x_c_char_train, vocab = \
            load_data(train_file, level=level)
        x_a_test, x_b_test, x_a_char_test, x_b_char_test, _ = load_data(test_file, level=level, test=True)
        return [x_a_train, x_b_train, x_c_train, x_a_char_train, x_b_char_train, x_c_char_train], \
               [x_a_test, x_b_test, x_a_char_test, x_b_char_test], vocab


def get_test_data(test_file=None, level='word'):
    if level == 'word':
        x_a_test, x_b_test, vocab = load_data(test_file, level=level, test=True)
        return [x_a_test, x_b_test], vocab
    else:
        x_a_test, x_b_test, x_a_char_test, x_b_char_test, vocab = load_data(test_file, level=level, test=True)
        return [x_a_test, x_b_test, x_a_char_test, x_b_char_test], vocab


class Train(object):
    def __init__(self, train_data, test_data, vocab, model_name='siamese_cnn'):
        self.vocab = vocab
        self.model_name = model_name
        self.get_config()
        self.get_model()
        if level == 'word':
            self.x_a_train, self.x_b_train, self.x_c_train = train_data
            self.x_a_test, self.x_b_test = test_data
            self.x_a_char_train, self.x_b_char_train, self.x_c_char_train = None, None, None
            self.x_a_char_test, self.x_b_char_test = None, None
        else:
            self.x_a_train, self.x_b_train, self.x_c_train, self.x_a_char_train, self.x_b_char_train, \
            self.x_c_char_train = train_data
            self.x_a_test, self.x_b_test, self.x_a_char_test, self.x_b_char_test = test_data

    def fold_train(self, fold=10):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=fold, random_state=7)
        cv = 1
        results_cv = []
        dev_results = []
        for train_index, valid_index in kf.split(self.x_a_train):
            self.get_model()
            if overwrite or not os.path.exists(
                    os.path.join(self.config.checkpoint_dir, '%s_%d.hdf5' % (self.config.exp_name, cv))):
                print('Start training the %s_%d_swa model...' % (self.config.exp_name, cv))
                max_acc = 0.
                no_more_num = 0
                for i in range(self.config.num_epochs):
                    print('epoch %d' % (i + 1))
                    self.siamese_model.fit(self.x_a_train, self.x_b_train, self.x_c_train, train_index, valid_index,
                                           x_a_char_train=self.x_a_char_train, x_b_char_train=self.x_b_char_train,
                                           x_c_char_train=self.x_c_char_train, swa=swa)
                    acc = self.siamese_model.predict_dev(self.x_a_train, self.x_b_train, self.x_c_train, valid_index,
                                                         a_char=self.x_a_char_train, b_char=self.x_b_char_train,
                                                         c_char=self.x_c_char_train)
                    if acc > max_acc:
                        no_more_num = 0
                        self.siamese_model.save_weights(cv)
                        max_acc = acc
                        continue
                    no_more_num += 1
                    print('max_acc:', max_acc, 'no_more_num:', no_more_num)
                    if no_more_num >= self.config.early_stopping_patience:
                        break
            self.siamese_model.load_weights(cv)
            dev_result = self.siamese_model.predict_dev(self.x_a_train, self.x_b_train, self.x_c_train, valid_index,
                                                        a_char=self.x_a_char_train, b_char=self.x_b_char_train,
                                                        c_char=self.x_c_char_train)
            dev_results.append(dev_result)
            results = self.siamese_model.predict(self.x_a_test, self.x_b_test, a_char=self.x_a_char_test,
                                                 b_char=self.x_b_char_test)
            self.siamese_model.evaluate(results)
            results_cv.append(results)
            if swa:
                self.siamese_model.load_swa_weight()
                print('Start evaluate the %s_%d_swa model...' % (self.config.exp_name, cv))
                results_swa = self.siamese_model.predict(self.x_a_test, self.x_b_test, a_char=self.x_a_char_test,
                                                         b_char=self.x_b_char_test)
                self.siamese_model.evaluate(results_swa)
            cv += 1
        if not os.path.exists('output/'):
            os.makedirs('output')
        self.save_results(dev_results)
        return results_cv

    def fold_merge(self):
        self.model_name = 'esim'
        dev_results = self.fold_train(10)
        self.model_name = 'siamese_cnn'
        dev_results.append(self.fold_train(10))
        self.write_results(dev_results)

    def write_results(self, results_cv):
        results_cv = np.asarray(results_cv)
        results = np.mean(results_cv, axis=0)
        print(results_cv.shape, results.shape)
        with codecs.open('output/output.txt', 'w', encoding='utf8') as f:
            b = 0.
            for i in range(results.shape[0]):
                if i % 2 == 0:
                    b = results[i]
                else:
                    c = results[i]
                    if b > c:
                        f.write("B\n")
                    else:
                        f.write("C\n")

    def save_results(self, dev_result):
        with codecs.open('log/'+self.model_name+'.txt', 'a', encoding='utf8') as f:
            f.write(self.model_name + "\n")
            for i in range(len(dev_result)):
                f.write(str(dev_result[i]) + '\n')

    def get_model(self):
        self.name2model()
        self.siamese_model = self.M(self.config)
        print('Create the %s model...' % self.config.exp_name)
        self.siamese_model.compile()

    def name2model(self):
        m = {'siamese_cnn': SiameseCNN,
             'esim': ESIM,
             'dpcnn': DPCNN}
        self.M = m[self.model_name]

    def get_config(self):
        self.config = Config()
        self.config.level = level
        self.config.max_len = self.config.max_len_word
        self.config.exp_name = self.model_name + '_' + level
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        if fasttext:
            self.config.embedding_file += 'fasttext'
            self.config.exp_name += '_fasttext'
        else:
            self.config.embedding_file += 'embeddings'


class Test(object):
    def __init__(self, test_data, vocab, model_name='siamese_cnn'):
        self.vocab = vocab
        self.model_name = model_name
        self.get_config()
        self.get_model()
        if level == 'word':
            self.x_a_test, self.x_b_test = test_data
            self.x_a_char_test, self.x_b_char_test = None, None
        else:
            self.x_a_test, self.x_b_test, self.x_a_char_test, self.x_b_char_test = test_data

    def get_model(self):
        self.name2model()
        self.siamese_model = self.M(self.config)
        print('Create the %s model...' % self.config.exp_name)
        self.siamese_model.compile()

    def name2model(self):
        m = {'siamese_cnn': SiameseCNN,
             'esim': ESIM,
             'dpcnn': DPCNN}
        self.M = m[self.model_name]

    def get_config(self):
        self.config = Config()
        self.config.level = level
        self.config.max_len = self.config.max_len_word
        self.config.exp_name = self.model_name + '_' + level
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        if fasttext:
            self.config.embedding_file += 'fasttext'
            self.config.exp_name += '_fasttext'
        else:
            self.config.embedding_file += 'embeddings'

    def fold_merge(self):
        self.model_name = 'esim'
        results_cv = self.fold_test(10)
        self.model_name = 'siamese_cnn'
        results_cv.append(self.fold_test(10))
        self.write_results(results_cv)

    def fold_test(self, fold=10):
        results_cv = []
        for i in range(fold):
            cv = i + 1
            self.siamese_model.load_weights(cv)
            results = self.siamese_model.predict(self.x_a_test, self.x_b_test, a_char=self.x_a_char_test,
                                                 b_char=self.x_b_char_test)
            results_cv.append(results)
        return results_cv

    def write_results(self, results_cv):
        results_cv = np.asarray(results_cv)
        results = np.mean(results_cv, axis=0)
        with codecs.open('/output/output.txt', 'w', encoding='utf8') as f:
            b = 0.
            for i in range(len(results)):
                if i % 2 == 0:
                    b = results[i]
                else:
                    c = results[i]
                    if b > c:
                        f.write("B\n")
                    else:
                        f.write("C\n")


if __name__ == '__main__':
    if train:
        train_data, test_data, vocab = get_data('data/input.txt', 'input/input.txt', level=level)
        train_cnn = Train(train_data, test_data, vocab, model_name='siamese_cnn')
        results = train_cnn.fold_train(10)
        train_esim = Train(train_data, test_data, vocab, model_name='esim')
        results += train_esim.fold_train(10)
        train_esim.write_results(results)
    else:
        test_data, vocab = get_test_data('/input/input.txt', level=level)
        test_esim = Test(test_data, vocab, model_name='esim')
        results = test_esim.fold_test(10)
        test_cnn = Test(test_data, vocab, model_name='siamese_cnn')
        results += test_esim.fold_test(10)
        test_cnn.write_results(results)
