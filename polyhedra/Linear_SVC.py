import numpy as np


class Linear_SVC(object):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, x, y):
        def loss():
            return sum(w_t * w_t) + self.C * sum(max(0, 1 - np.dot(w_t, xi) * yi) for xi, yi in zip(x, y))

        x = np.array(x)
        self.labels = y[0], tuple(label for label in y if label != y[0])[0]
        y = np.array(tuple(-1 if label == y[0] else 1 for label in y), dtype='int32')

        opt_dict = {}
        transforms = (1, 1), (-1, 1), (-1, -1), (1, -1)
        self.max_feature_value = max(x.flatten())
        self.min_feature_value = min(x.flatten())

        step_sizes = (  # with smaller steps our margins and db will be more precise
            self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001)  # point of expense

        b_range_multiple = 1  # this parameter increase accuracy but very expensive
        b_multiple = 200  # this parameter makes fitting faster sacrificing accuracy
        latest_optimum = self.max_feature_value * 10

        # objective is to satisfy yi(x.w) + b >= 1
        # for all training dataset such that loss is minimum
        # for this we will use loss function

        for step in step_sizes:  # making step smaller and smaller to get precise value
            w = np.array((latest_optimum, latest_optimum), dtype='float32')
            optimized = False
            while not optimized:
                for b in np.arange(-self.max_feature_value * b_range_multiple, self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for xi, yi in zip(x, y):
                            if not yi * (sum(w_t * xi) + b) >= 1:
                                found_option = False
                                break
                        if found_option:
                            # all points in dataset satisfy y(w.x) + b >= 1 for this cuurent w_t, b
                            # then put w, b in dict with loss as key
                            opt_dict[loss()] = w_t, b
                if w[0] < 0:
                    optimized = True
                else:
                    w -= step

            self.w, self.b = opt_dict[min(opt_dict)]  # optimal values of w, b
            # start with new latest_optimum (initial values for w)
            latest_optimum = self.w[0] + step * 2

    def predict(self, features):
        return np.array(tuple(self.labels[int(prediction)] for prediction in np.sign(np.dot(np.array(features), self.w) + self.b) > 0))
