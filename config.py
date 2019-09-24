# -*- coding: utf-8 -*-


class Config(object):
    def __init__(self):
        self.level = "word"
        self.checkpoint_dir = 'models'
        self.exp_name = None
        self.embedding_path = None
        self.embedding_path_word = None
        self.embedding_path_char = None
        self.max_len = None
        self.vocab_len = None
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.optimizer = "adam"
        self.batch_size = 64  # preattcnn 20
        self.verbose_training = 1
        self.checkpoint_monitor = "val_acc"
        self.checkpoint_mode = "max"
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_verbose = True
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_patience = 20
        self.early_stopping_mode = 'max'
        self.max_len_word = 500
        self.max_len_char = 300
        self.vocab_len_word = 4536
        self.vocab_len_char = 1541
        self.char_per_word = 5
        self.embedding_path = "data"
        self.embedding_file = 'cail_300_dim.'
        self.embedding_dim = 300
        self.margin = 0.15
        self.dropout = 0.2