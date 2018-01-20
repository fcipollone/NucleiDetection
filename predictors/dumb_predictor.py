import numpy as np


class DumbPredictor:
    def __init__(self, corpus, corpus_test=None):
        self.corpus = corpus
        self.corpus_test = corpus

    def train(self):
        train = self.corpus.get_list_images_and_masks()
        do_something = 0

    def predict(self):
        for example in self.corpus_test.get_examples():
            prediction = np.zeros_like(example.image)
            example.set_prediction(prediction)

    def write(self, filename):
        self.corpus_test.generate_submission(filename)