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
f_prefx = '/data/inphase_quadrature_'
with open(f_path + f_prefx + 'data.json') as f:
    iqdata = np.array(json.load(f))
with open(f_path + f_prefx + 'noise.json') as f:
    iqnoise = np.array(json.load(f))
# Convert to the complex row vectors
iqdata = iqdata[:, 0, :] + 1j * iqdata[:, 1, :]
iqnoise = iqnoise[:, 0, :] + 1j * iqnoise[:, 1, :]

dbasis = get_clusters()
# print(dbasis)
cls_sizes = [0, 0, 0, 10]
dset = gen_base(iqdata, iqnoise, dbasis, cls_sizes)
print(list(dset[0].keys()))
# print(dset[0])

answer_y = 7

y = dset[answer_y]['data']
answer_X = dset[answer_y]['basis']
answer_coeff = dset[answer_y]['coeff']
answer_shift = dset[answer_y]['shift']
print(answer_X)

#plt_clust_Xy(iqdata[answer_X], y, y, answer_y, 0)
data_scaled = scale_complex(iqdata[answer_X], answer_coeff)
plt_clust_Xyy(data_scaled, y, y, answer_X, answer_y, 0)

n_basis = len(dbasis)  # 64
idx_A = list(dbasis.keys())
AT = iqdata[idx_A]
A = AT.transpose()

mdl = {frozenset(): {'err': np.inf,
                     'fea': list([]),  # features in the right order
                     'sft': list([]),  # shift for each feature
                     'par': np.empty([])  # set of parameters for each feature
                     }
       }  # Empty dictionary of {indices: error}
#par_lst = err_lst.copy()  # {indices: parameters} shift for each column in the basis
mdl_new = mdl.copy()  # Updated dictionary

cnt = 0
MAX_SHIFT = 9
MAX_BASIS = 5
MAX_MODELS = 200
while cnt < MAX_BASIS:
    cnt += 1
    for idx in mdl.keys():
        # print('Append to the indices:', idx)
        for j in range(n_basis):  # Exhaustive search in the set of all features of A
            # Append indices of the features
            idx_new = frozenset(set(idx) | {j})
            if idx_new in mdl.keys():  # Keep all history in mdl_new
                # print('Already in the dictionary:', idx_cp)
                continue
            # Compute the error and add it to the dictionary
            x = A[:, j]
            if not idx:
                X = np.array([])
                shifts = []
                features = []
            else:
                features = mdl[idx]['fea'].copy()
                shifts = mdl[idx]['sft'].copy()
                X = A[:, features]
                X = shift(X, shifts)  # Arrange the shift
            err_min, best_b, best_shift = find_shiftX_exhaust(X, x, y, MAX_SHIFT)
            shifts.append(best_shift)
            features.append(j)
            # best_b needs no append sine the last feature corresponds to the last item in this vector
            mdl_new.update({idx_new: {'err': err_min, 'fea': features, 'sft': shifts, 'par': best_b}})
            # {data[i] for i in indexes}
            # print('idx=', idx_new, 'err=', err_min, 'fea=', features, 'sft=', shifts, 'par=', best_b)
            if cnt == MAX_BASIS:
                if set(answer_X) == {idx_A[i] for i in features}:
                    print('hit to the answer', 'answer_X =', set(answer_X), 'idx_A =', {idx_A[i] for i in features})
                    X = A[:, features]
                    X1 = shift(X, shifts)
                    # best_b1 = best_b[::-1]
                    y1 = X1 @ best_b
                    plt_clust_Xyy(X1.transpose(), y, y1, answer_X, answer_y, 0)

    mdl = dict(sorted(mdl_new.items(), key=lambda i: i[1]['err'])[:MAX_MODELS])
#--- end forfor





cnt = 0
for idx in mdl:
    cnt += 1
    X = A[:, list(idx)]
    err = mdl[idx]['err']
    par = mdl[idx]['par']
    sft = mdl[idx]['sft']
    X1 = shift(X, sft)
    y1 = X1 @ par
    # plt_clust_Xyy(X1.transpose(), y, y1, list(idx), 'E {:.3f}'.format(err), cnt)
