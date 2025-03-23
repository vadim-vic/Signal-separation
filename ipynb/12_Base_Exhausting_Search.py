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
del f, f_path, f_prefix

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

    def plot_mdl(self, y, max_models=None):
        # Plot all
        if max_models is None:
            max_models = len(self.mdl) # Plots all items from mdl

        for idx, cnt in zip(fs.mdl, range(len(self.mdl))):
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
# Load the basis features
dbasis = get_clusters() # The key is the centroid index, the value is the vector of item indices

idx_basis = list(dbasis.keys())
Abase = iqdata[idx_basis].transpose()

# Load the data
cls_sizes = [0, 0, 0, 0, 0, 100]  # Set sample size for i-th collision
dset = gen_base(iqdata, iqnoise, dbasis, cls_sizes)
print(list(dset[0].keys())) # List of the keys in each data sample

n_models = 6
n_classes = len(cls_sizes)
# Later we check the reconstructed signal with one of the cluster's signals

# Set one data sample to reconstruct
def next_sample_dset(dset, idx_basis):
    # Yields the next sample in the dset item every time it's called
    # The structure of the dictionary dset is in the function gen_base
    for idx in range(len(dset)):
        answer_y = dset[idx]['data']
        answer_X = dset[idx]['basis']
        answer_A = [idx_basis.index(i) if i in idx_basis else -1 for i in answer_X]  # What if -1 happens
        answer_coeff = dset[idx]['coeff']
        answer_shift = dset[idx]['shift']
        yield answer_y, answer_X, answer_A, answer_coeff, answer_shift
next_sample = next_sample_dset(dset, idx_basis)

# answer_y, answer_X, answer_A, answer_coeff, answer_shift = next(next_sample)
# dscaled = scale_complex(iqdata[answer_X], answer_coeff)
# plt_clust_Xyy(dscaled, answer_y, answer_y, answer_X, 0, 0)
# del dscaled, answer_X

# Check the reconstruction quality
cnt_err = 0
cnt_resolved = 0
X_cls = np.empty([0,n_models * n_classes])  # objects to 4 class classification
y_cls = []  # target 4 classes
for i in range(100):
    # Get a sample from the dataset
    answer_y, answer_X, answer_A, answer_coeff, answer_shift = next(next_sample)
    fs = FeatureSelection(Abase, max_models = n_models ) # Reset the list of models
    # Run the reconstruction procedure
    # fs.mdl = run(fs.A, answer_y, fs.mdl, max_basis=5, max_models=6)

    i_dist = np.array([])
    # For each number of collided signals
    for c in range(n_classes):
        fs.mdl = run(fs.A, answer_y, fs.mdl, max_basis = 1,  max_models = n_models)
        i_dist = np.hstack((i_dist, np.array([values['err'] for values in fs.mdl.values()])))
        # print(np.shape(i_dist))
    #if np.size(X_cls) == 0:
    #    X_cls = np.array([i_dist])
    #else:
    X_cls = np.vstack((X_cls, i_dist.transpose()))
    y_cls.append(len(answer_A))
    # Plot the best model
    # fs.plot_mdl(answer_y, 1)
    # Check the quality of the reconstruction by comparing with the answer
    best_A = fs.best_model()['fea']

    dist = fs.best_model()['err']
    if set(answer_A) == set(best_A):
        print(i, 'errors:', dist)
    else:
        cnt_err += 1
        print(i, 'errors:', dist, 'answer:', answer_A, 'model:', best_A)
    len_resolved = len(set(answer_A) & set(best_A))
    if len_resolved > 0:
        cnt_resolved += 1
        print(i, 'resolved:', len_resolved)
print('_____________________')
print(cnt_err , cnt_resolved)
