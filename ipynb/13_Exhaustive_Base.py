# Find the base for various number of mixture signals. Add the alternative base features sequentially.
# Select them according to the approximation to the external error.

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from dataload import (load_data, get_clusters, gen_base, next_sample_dset)
from experiment import FeatureSelection
from plots import plt_lines

# Load data and data basis
iqdata, iqnoise = load_data()
dbasis = get_clusters()

# Create a data generator and the main class instance
# The indices of base centroids in iqdata
idx_Abase = list(dbasis.keys())
Abase = iqdata[idx_Abase].transpose()

# How to use the data:
# Generate new dset with randomly mixed signals as y plus a noise of desired level
class_sizes = [3, 3, 3, 3, 3, 3]  # Set sample size for each collided groups
# dset = gen_base(iqdata, iqnoise, dbasis, cls_sizes)
# print(list(dset[0].keys())) # The keys in each data sample

# This instance delivers a new y with answers
# get_next_sample = next_sample_dset(dset, idx_Abase) # Use after generation new dset
# This instance carry models for each y
# fs = FeatureSelection(Abase) # Use after get a new y

# The parameters of the computational experiment
# n_models = 6`
# n_classes = len(class_sizes)
# Later we check the reconstructed signal with one of the cluster's signals
# Check the reconstruction quality
# create the table to plot
n_noise_levels = 10 # r, rows
n_classes = len(class_sizes) # c, columns
noise_levels = np.linspace(0, 1, n_noise_levels)
cnt_resolvedA = np.zeros((n_noise_levels, n_classes))
cnt_resolved1 = cnt_resolvedA.copy()
cnt_resolved2 = cnt_resolvedA.copy()

n_samples = 30

for noise_level, r in zip(noise_levels, range(n_noise_levels)):
    for c in range(1,n_classes):

        class_sizes =  np.zeros(len(class_sizes), dtype=int)
        class_sizes[c] = n_samples
        # Generate the new noisy dset of mixtures and create a data sample generator
        # dset = get_dset(class_sizes, noise_level)
        dset = gen_base(iqdata, iqnoise, dbasis, cls_sizes=class_sizes, noise_level=noise_level)
        get_next_sample = next_sample_dset(dset, idx_Abase)

        i_resolvedA, i_resolved1, i_resolved2 = 0, 0, 0
        for i in range(n_samples):
            answer_y, answer_A, _, _, _ = next(get_next_sample)
            fs = FeatureSelection(Abase, max_models = 6) # Reset the list of models
            fs.MAX_BASIS = c
            fs.run(answer_y)
            best_A = fs.best_model()['fea']
            error = fs.best_model()['err']
            if set(answer_A) == set(best_A):
                i_resolvedA += 1
            len_resolved = len(set(answer_A) & set(best_A))
            if len_resolved > 1:
                i_resolved1 += 1
            if len_resolved > 2:
                i_resolved2 += 1

            print('rc', r,c, 'i', i, '12A:', i_resolved1, i_resolved2, i_resolvedA,  'answer:', answer_A, 'model:', best_A, 'err:', f'{ error:.2f}')
        cnt_resolvedA[r,c] = i_resolvedA
        cnt_resolved1[r,c] = i_resolved1
        cnt_resolved2[r,c] = i_resolved2

_ = 100 * cnt_resolved1[:, 1:] / n_samples
plt_lines(_, noise_levels, 'one')
_ = 100 * cnt_resolved2[:, 1:] / n_samples
plt_lines(_, noise_levels, 'two')
_ = 100 * cnt_resolvedA[:, 1:] / n_samples
plt_lines(_, noise_levels, 'all')

np.save('../tmp/cnt_resolved1.npy', cnt_resolved1)
np.save('../tmp/cnt_resolved2.npy', cnt_resolved2)
np.save('../tmp/cnt_resolvedA.npy', cnt_resolvedA)
