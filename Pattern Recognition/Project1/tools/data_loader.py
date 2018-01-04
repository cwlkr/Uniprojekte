import csv

import numpy as np


class DataLoader:

    @staticmethod
    def load(filename, limit=None):
        data = []
        labels = []

        with open(filename, newline='') as csvfile:
            for i, row in enumerate(csv.reader(csvfile, delimiter=',')):
                labels.append(int(row.pop(0)))
                data.append(np.array(row, dtype=np.float64))

                if limit and i > limit:
                    break

        return (data, labels)
