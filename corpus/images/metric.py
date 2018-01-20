import numpy as np


class Metric:
    def __init__(self, prediction, mask):
        self.prediction = prediction
        self.mask = mask

    def rename_island(self, bin_map, i, j, val):
        if i < 0 or j < 0 or i >= bin_map.shape[0] or j >= bin_map.shape[1]:
            return bin_map
        if bin_map[i][j] == 1:
            bin_map[i][j] = val
            for new_i in [-1, 0, 1]:
                for new_j in [-1, 0, 1]:
                    if new_j == new_i == 0:
                        continue
                    if new_i != 0 and new_j != 0:
                        continue
                    bin_map = self.rename_island(bin_map, i+new_i, j+new_j, val)
        return bin_map

    def rename_points(self, bin_map):
        bin_map = bin_map.astype(int)
        cur_island = 2
        for i in range(bin_map.shape[0]):
            for j in range(bin_map.shape[1]):
                if bin_map[i][j] == 1:
                    bin_map = self.rename_island(bin_map, i, j, cur_island)
                    cur_island += 1
        print(cur_island)
        #import pdb; pdb.set_trace()
        bin_map -= 1
        bin_map[bin_map == -1] = 0
        return bin_map

    # Precision helper function
    def precision_at(self, threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    def compute_metric(self):
        #import pdb; pdb.set_trace()
        labels = self.rename_points(self.mask)
        y_pred = self.rename_points(self.prediction)
        self.labels = labels
        self.y_pred = y_pred
        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)
        union = area_true + area_pred - intersection
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Loop over IoU thresholds
        prec = []
        print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = self.precision_at(t, iou)
            p = tp / (tp + fp + fn)
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
