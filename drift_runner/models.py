class MajorityClassModel:
    def __init__(self):
        self.class_counts = {}

    def predict_one(self, x):
        if not self.class_counts:
            return None
        return max(self.class_counts, key=self.class_counts.get)

    def learn_one(self, x, y):
        self.class_counts[y] = self.class_counts.get(y, 0) + 1


class NoChangeModel:
    def __init__(self):
        self.last_label = None

    def predict_one(self, x):
        return self.last_label

    def learn_one(self, x, y):
        self.last_label = y