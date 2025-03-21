# General import
import numpy as np
import matplotlib.pyplot as plt
import json
# Import the local functions, plots, and utilities
from functions import find_shiftX_exhaust, shift, scale_complex
from functions import gen_base, get_clusters, shift_x, is_incluster
from plots import plt_clust_Xy, plt_clust_Xyy

# Read the data flies and import functions
f_path = '/Users/victor/PycharmProjects/Signal-separation/'
f_prefix = '/data/inphase_quadrature_'
# Convert to the complex row vectors
with open(f_path + f_prefix + 'data.json') as f:
    iqdata = np.array(json.load(f))
    iqdata = iqdata[:, 0, :] + 1j * iqdata[:, 1, :]
with open(f_path + f_prefix + 'noise.json') as f:
    iqnoise = np.array(json.load(f))
    iqnoise = iqnoise[:, 0, :] + 1j * iqnoise[:, 1, :]

def run(A, y, mdl, max_basis = 1, max_models = 100, max_shift = 9):
    # Exhaustive search of linear combination of columns A to approximate y
    # A is a (m,n) matrix with n basis features of length m
    # y is unknown linear combination of small number of features from A of length m
    # mdl is a special format dictionary with the model parameters
    # it contains various linear combinations an alternative to approximate
    # max_basis in the number of features to append to the linear combination
    # max_models restricts number of the alternative models to avoid exp growth
    # Returns: updated mdl
    cnt_basis = 0
    n_basis = np.size(A,1)
    mdl_new = mdl.copy()
    while cnt_basis < max_basis:
        cnt_basis += 1
        for idx in mdl.keys():
            # print('Append to the indices:', idx)
            for j in range(n_basis):  # Exhaustive search in the set of all features of A
                # Append index of a feature to the new dictionary
                idx_new = frozenset(set(idx) | {j})
                if idx_new in mdl: continue # Already in the dictionary, drop it
                # Compute the error, parameters and add them to the dictionary values
                x = A[:, j]
                # Append the new feature to the existing matrix
                features = mdl[idx]['fea'].copy()
                shifts = mdl[idx]['sft'].copy()
                X = A[:, features]
                X = shift(X, shifts)  # Arrange the shift
                err_min, best_b, best_shift = find_shiftX_exhaust(X, x, y, max_shift)
                features.append(j)
                shifts.append(best_shift)
                # In the vector best_b the last feature corresponds to the last item
                mdl_new.update({idx_new: {'err': err_min, 'fea': features, 'sft': shifts, 'par': best_b}})

    mdl = dict(sorted(mdl_new.items(), key=lambda i: i[1]['err'])[:max_models])
    return mdl

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

    def plot_mdl(self, max_models=None):
        # Plot all
        if max_models is None:
            max_models = len(self.mdl) # Plots all items from mdl

        for idx, cnt in zip(fs.mdl, range(len(self.mdl))):
            X = self.A[:, list(idx)]
            err = self.mdl[idx]['err']
            par = self.mdl[idx]['par']
            sft = self.mdl[idx]['sft']
            X = shift(X, sft)
            y1 = X @ par
            plt_clust_Xyy(X.transpose(), y, y1, list(idx), 'E {:.3f}'.format(err), cnt)
            if cnt > max_models:
                break


dbasis = get_clusters() # The key is the centroid index, the value is the vector of item indices
# Later we check the reconstructed signal with one of the cluster's signals

cls_sizes = [0, 0, 0, 10]  # Set sample size for i-th collision
dset = gen_base(iqdata, iqnoise, dbasis, cls_sizes)
print(list(dset[0].keys())) # List of the keys in each data sample
# print(dset[0])

# Set one data sample to reconstruct
answer_y = 7
y = dset[answer_y]['data']
answer_X = dset[answer_y]['basis']
answer_coeff = dset[answer_y]['coeff']
answer_shift = dset[answer_y]['shift']
print(answer_X)

data_scaled = scale_complex(iqdata[answer_X], answer_coeff)
plt_clust_Xyy(data_scaled, y, y, answer_X, answer_y, 0)

del data_scaled, cls_sizes, dset

A = iqdata[list(dbasis.keys())].transpose()
fs = FeatureSelection(A,y)
fs.mdl = run(fs.A, y, fs.mdl, max_basis = 1)
fs.plot_mdl()
