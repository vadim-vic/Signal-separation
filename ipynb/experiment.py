# This modile contains only one class to carry the main compurational experiment
import numpy as np
from functions import find_shiftX_exhaust, shift
from functions import max_weighted_distance
from plots import plt_clust_Xyy

class FeatureSelection:
    # Collects and analyses basis feature set to approximate unknown linear combination
    def __init__(self, A, max_shift=9, max_basis=5, max_models=100):
        self.A = A
        self.n_basis = np.size(A,1) # Number of columns
        # self.answer_X = answer_X
        self.MAX_SHIFT = max_shift
        self.MAX_BASIS = max_basis
        self.MAX_MODELS = max_models

        self.mdl = {frozenset(): {'err': np.inf,  # key is the features
                             'fea': list([]),  # features in the right order
                             'sft': list([]),  # shift for each feature, same order
                             'par': np.empty([])  # set of parameters for each feature
                             }}

    def run(self, y):
        # Exhaustive search of linear combination of columns A to approximate y
        # A is a (m,n) matrix with n basis features of length m
        # y is unknown linear combination of small number of features from A of length m
        # mdl is a special format dictionary with the model parameters
        # it contains various linear combinations an alternative to approximate
        # max_basis in the number of features to append to the linear combination
        # max_models restricts number of the alternative models to avoid exp growth
        # Returns: updated mdl
        # From the previous version: A, mdl, max_basis max_models max_shift
        cnt_basis = 0
        mdl_new = self.mdl.copy()
        while cnt_basis < self.MAX_BASIS:
            cnt_basis += 1
            for idx in self.mdl.keys():
                # print('Append to the indices:', idx)
                for j in range(self.n_basis):  # Exhaustive search in the set of all features of A
                    # Append index of a feature to the new dictionary
                    idx_new = frozenset(set(idx) | {j})
                    if idx_new in self.mdl: continue  # Already in the dictionary, drop it
                    # Compute the error, parameters and add them to the dictionary values
                    x = self.A[:, j]
                    # Append the new feature to the existing matrix
                    features = self.mdl[idx]['fea'].copy()
                    shifts = self.mdl[idx]['sft'].copy()
                    X = self.A[:, features]
                    X = shift(X, shifts)  # Arrange the shift
                    err_min, best_b, best_shift = find_shiftX_exhaust(X, x, y, self.MAX_SHIFT)
                    features.append(j)
                    shifts.append(best_shift)
                    # In the vector best_b the last feature corresponds to the last item
                    mdl_new.update({idx_new: {'err': err_min, 'fea': features, 'sft': shifts, 'par': best_b}})
            # Choose carefully the error function before the best model selection
            # self.update_error(y) # If you need an alternative error function to  select models
            # Select the best max_models with the minimum error
            self.mdl = dict(sorted(mdl_new.items(), key=lambda i: i[1]['err'])[:self.MAX_MODELS])

    def plot_mdl(self, y, max_models=None):
        # Plot all
        if max_models is None:
            max_models = len(self.mdl) # Plots all items from mdl

        for idx, cnt in zip(self.mdl, range(len(self.mdl))):
            X = self.A[:, list(idx)]
            fea = self.mdl[idx]['fea']
            err = self.mdl[idx]['err']
            par = self.mdl[idx]['par']
            sft = self.mdl[idx]['sft']
            X = shift(X, sft)
            y1 = X @ par
            plt_clust_Xyy(X.transpose(), y, y1, fea, 'E {:.3f}'.format(err), cnt)
            if cnt > max_models:
                break

    def best_model(self):
        best_mdl = min(self.mdl.items(), key=lambda i: i[1]['err'])[1]
        return best_mdl

    def update_error(self, y):
        # The Hausdorff distance is used here to replace the Euclidean error.
        for idx in self.mdl.keys():
            fea = self.mdl[idx]['fea']
            sft = self.mdl[idx]['sft']
            par = self.mdl[idx]['par']
            X = self.A[:, fea]
            X = shift(X, sft)
            y1 = X @ par
            error = max_weighted_distance(y, y1)
            self.mdl[idx]['err'] = error
