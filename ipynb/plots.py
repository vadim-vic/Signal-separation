# * Collection of plot functions for the I/Q data signal visualization
# The singal dtype=complex
import matplotlib.pyplot as plt
import numpy as np

def plt_clust_Xyy(X, y, y1, idx_X, idx_y, cnt):
# Plot (separated real, imag plot) all signals of the matrix X as a cluster
# and the signal y over it as either a centriod or a target.
  if len(X.shape) == 1:
    X = np.column_stack([X])

  # plt.rcParams['text.usetex'] = True
  plt.rcParams['font.family'] = 'DejaVu Serif'
  plt.rcParams['lines.linewidth'] = 2
  plt.rcParams['xtick.labelsize'] = 12#24
  plt.rcParams['ytick.labelsize'] = 12#24
  plt.rcParams['legend.fontsize'] = 12#24
  plt.rcParams['axes.labelsize'] = 10#24
  # Create a figure and two subplots (1 row, 2 columns)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

  idx_X = list(range(X.shape[0])) if idx_X is None else idx_X
  idx_y = -1 if idx_y is None else  idx_y

  for i in range(len(idx_X)):
    ax1.plot(X.real[i], label=str(idx_X[i]))
    ax2.plot(X.imag[i], label=str(idx_X[i]))
  # Finalize the plot (real)
  ax1.plot(y.real, label=str(idx_y), color='black', linewidth=2)
  ax1.plot(y1.real, color='red', linewidth=2)
  ax1.set_ylabel('Amplitude, V (real)', fontname='DejaVu Serif')
  ax1.set_xlabel('Time ticks', fontname='DejaVu Serif')
  # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
  # Finalize the plot (imaginarty)
  ax2.plot(y.imag, label=str(idx_y), color='black', linewidth=2)
  ax2.plot(y1.imag, color='red', linewidth=2)
  ax2.set_ylabel('Amplitude, V (imaginary)', fontname='DejaVu Serif')
  ax2.set_xlabel('Time ticks', fontname='DejaVu Serif')
  ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
  # The whole plot
  plt.tight_layout()
  plt.savefig('../tmp/' + str(cnt) + '_' + str(idx_y) + str(idx_X) + '_imag.png', dpi=300, bbox_inches='tight')
  plt.close()
  return

def plt_clust_Xy(X, y, idx_X = None, idx_y = None):
# Plot (separated real, imag plot) all signals of the matrix X as a cluster
# and the signal y over it as either a centriod or a target.
  if len(X.shape) == 1:
    X = np.column_stack([X])

  # plt.rcParams['text.usetex'] = True
  plt.rcParams['font.family'] = 'DejaVu Serif'
  plt.rcParams['lines.linewidth'] = 2
  plt.rcParams['xtick.labelsize'] = 12#24
  plt.rcParams['ytick.labelsize'] = 12#24
  plt.rcParams['legend.fontsize'] = 12#24
  plt.rcParams['axes.labelsize'] = 10#24

  idx_X = list(range(X.shape[0])) if idx_X is None else idx_X
  idx_y = -1 if idx_y is None else  idx_y

  for i in range(len(idx_X)):
    plt.plot(X.real[i], label = str(idx_X[i]))
  plt.plot(y.real, label = str(idx_y), color='black', linewidth=3)

  # Finalize the plot (real)
  plt.ylabel('Amplitude, V (real)', fontname='DejaVu Serif')
  plt.xlabel('Time ticks', fontname='DejaVu Serif')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.savefig('../tmp/' + str(idx_y[0])+'_real.png', dpi=300, bbox_inches='tight')
  plt.close()

  for i in range(len(idx_X)):
    plt.plot(X.imag[i], label = str(idx_X[i]))
  plt.plot(y.imag, label = str(idx_y), color='black', linewidth=3)

  # Finalize the plot (imaginarty)
  plt.ylabel('Amplitude, V (imaginary)', fontname='DejaVu Serif')
  plt.xlabel('Time ticks', fontname='DejaVu Serif')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.savefig('../tmp/' + str(idx_y[0])+'_imag.png', dpi=300, bbox_inches='tight')
  plt.close()
  return

# ** Plot (separated real, imag plot) all signals a the matrix as a cluster
def plt_cluster(X, idx): # X and X1 must be of the same shape
  #indices = np.random.choice(len(X), n, replace=False)# [0] # Pick an item
  #plt.rcParams['text.usetex'] = True
  plt.rcParams['font.family'] = 'DejaVu Serif'
  plt.rcParams['lines.linewidth'] = 2
  #plt.rcParams['lines.markersize'] = 12
  plt.rcParams['xtick.labelsize'] = 12#24
  plt.rcParams['ytick.labelsize'] = 12#24
  plt.rcParams['legend.fontsize'] = 12#24
  plt.rcParams['axes.labelsize'] = 10#24
  for i in idx:
    plt.plot(X.real[i], label = str(i))#, linestyle='dashed')
  plt.ylabel('Amplitude, V (real)', fontname='DejaVu Serif')
  plt.xlabel('Time ticks', fontname='DejaVu Serif')
  plt.legend()
  plt.show()
  for i in idx:
    plt.plot(X.imag[i], label = str(i))#, linestyle='dashed')
    plt.ylabel('Amplitude, V (imaginary)', fontname='DejaVu Serif')
    plt.xlabel('Time ticks', fontname='DejaVu Serif')
    plt.legend()
  plt.show()
  return
  
# ** Plot (single plot) a couple of signals as vectors and a couple of signals from a matrix
def plt_compare_vec(x, y, xlabel = 'First', ylabel = 'Second'): # X and X1 must be of the same shape
  #indices = np.random.choice(len(X), n, replace=False)# [0] # Pick an item
  #for index in indices:
  plt.plot(x.real, label = 'First, re', linestyle='dashed')
  plt.plot(x.imag, label = 'First, im', linestyle='dashed')
  plt.plot(y.real, label = 'Second, re')
  plt.plot(y.imag, label = 'Second, im')
  plt.rcParams['font.family'] = 'DejaVu Serif'
  plt.rcParams['lines.linewidth'] = 2
  #plt.rcParams['lines.markersize'] = 12
  plt.rcParams['xtick.labelsize'] = 12#24
  plt.rcParams['ytick.labelsize'] = 12#24
  plt.rcParams['legend.fontsize'] = 12#24
  plt.rcParams['axes.labelsize'] = 10#24
  plt.xlabel('Time ticks', fontname='DejaVu Serif')
  plt.ylabel('Amplitude, V', fontname='DejaVu Serif')
  plt.legend()
  plt.show()
  return
  
  
def plot_AUC(auc, accuracy, fpr, tpr):
  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
  plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
  plt.rcParams['font.family'] = 'DejaVu Serif'
  plt.rcParams['lines.linewidth'] = 2
  #plt.rcParams['lines.markersize'] = 12
  plt.rcParams['xtick.labelsize'] = 12#24
  plt.rcParams['ytick.labelsize'] = 12#24
  plt.rcParams['legend.fontsize'] = 12#24
  #plt.rcParams['axes.titlesize'] = 36.
  plt.rcParams['axes.labelsize'] = 12#24
  plt.gca().set_aspect('equal')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.tight_layout()
  plt.legend(loc='lower right')
  plt.show()
  return