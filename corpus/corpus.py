class Corpus:
    def __init__(self, examples):
        self.examples = examples

    def generate_submission(self, filename):
        f = open(filename, 'w')
        f.write('ImageId,EncodedPixels\n')
        for el in (self.examples):
            f.write(el.get_csv_line())
            f.write('\n')
        f.close()

    def get_list_images(self):
        return [ex.image for ex in self.examples]

    def get_list_images_and_masks(self):
        return [(ex.image, ex.mask) for ex in self.examples]

    def get_examples(self):
        return self.examples

