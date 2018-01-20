class Corpus:
    def __init__(self, examples):
        self.examples = examples

    def generate_submission(self, filename):
        f = open('filename','w')
        f.write('ImageId,EncodedPixels\n')
        for el in range(len(self.examples)):
            f.write(el.get_csv_line())
            f.write('\n')
        f.close()
