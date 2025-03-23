# * Collection of functions for I/D data signal processing
# The signal dtype=complex
import numpy as np


def get_clusters():
    idx_clusters = {(762, 647, 7, 169, 794, 331, 886, 183, 538, 859),
                    (195, 612, 133, 325, 40, 136, 733, 171, 844, 936, 494, 463, 439, 88, 985, 955, 189, 734),
                    (385, 738, 840, 112, 146, 530, 831), (224, 800, 5, 281, 517, 711, 9, 586, 842, 535, 152, 121, 862),
                    (777, 267, 664, 555, 815, 566, 185, 698, 709, 837, 456, 594, 731, 860, 606, 996, 358, 619, 627), (
                        514, 645, 775, 264, 265, 393, 520, 791, 413, 286, 543, 38, 294, 309, 949, 316, 318, 726, 987,
                        997,
                        511),
                    (471, 578, 681, 28, 527, 498, 403, 468, 277, 245, 438, 377, 220, 319),
                    (577, 865, 259, 452, 389, 361, 969, 523, 273, 595, 767, 952, 797, 191),
                    (321, 162, 994, 356, 774, 295, 491, 269, 909, 368, 785, 434, 499, 881),
                    (672, 100, 613, 932, 71, 487, 970, 303, 274, 658, 729),
                    (384, 352, 552, 876, 973, 696, 728, 766, 692, 820, 792, 61, 30, 447),
                    (513, 398, 787, 409, 921, 174, 433, 563, 53, 450, 454, 332, 591, 83, 485, 877, 878, 892, 637, 254,
                     383),
                    (802, 771, 901, 744, 600, 683, 461, 366, 976, 529, 338, 536, 890, 700, 829, 510, 415),
                    (386, 29, 35, 678, 42, 568, 317, 832, 327, 713, 216, 990, 350, 223, 353, 102, 753, 246, 247),
                    (963, 682, 107, 251, 655, 50, 984, 187),
                    (544, 928, 34, 549, 679, 967, 459, 588, 972, 814, 855, 80, 951, 818, 308, 663, 57),
                    (65, 741, 262, 614, 235, 587, 748, 430, 558, 114, 179, 978, 918, 151, 694, 506, 95),
                    (993, 515, 231, 810, 906, 782, 175, 560, 783, 210, 371, 853, 182, 375, 636, 445), (
                        641, 905, 525, 142, 147, 659, 427, 428, 812, 305, 823, 184, 314, 571, 194, 581, 843, 854, 215,
                        605,
                        868,
                        364, 751, 382), (899, 67, 44, 78, 703, 84, 666, 891, 285, 287),
                    (609, 418, 930, 518, 712, 590, 654, 910, 948, 373, 917, 408, 249, 953, 988, 893, 894),
                    (354, 706, 132, 451, 137, 10, 299, 938, 334, 888, 633, 186, 155, 607),
                    (387, 400, 21, 407, 291, 807, 553, 178, 306, 59, 63, 82, 470, 856, 94, 228, 374, 889, 124),
                    (288, 481, 98, 992, 36, 838, 871, 424, 302, 432, 592, 691, 924),
                    (394, 279, 923, 806, 935, 688, 188, 958, 960, 458, 718, 977, 339, 85, 597, 347, 242, 634, 127),
                    (737, 37, 550, 455, 202, 460, 653, 749, 781, 49, 379, 276, 437, 983, 158, 765, 222, 763),
                    (1, 97, 324, 968, 201, 149, 213, 981, 539),
                    (3, 422, 839, 72, 939, 473, 14, 48, 466, 819, 55, 217, 764),
                    (225, 257, 995, 134, 742, 870, 173, 975, 528, 278, 758, 24, 346, 60, 799),
                    (675, 292, 869, 599, 359, 8, 232, 297, 426, 908, 813, 965, 982, 572, 476, 828, 509, 62),
                    (256, 322, 453, 329, 496, 848, 370, 340, 148, 312, 478),
                    (611, 70, 326, 616, 234, 492, 205, 46, 493, 752, 52, 405, 150, 825, 601, 827, 93),
                    (416, 768, 962, 548, 168, 618, 811, 301, 141, 944, 884, 280, 477),
                    (545, 580, 484, 646, 39, 746, 238, 367, 850, 501, 214, 760, 25),
                    (128, 769, 803, 101, 841, 524, 333, 465, 497, 562, 979, 502, 695, 313, 858, 735),
                    (954, 673, 804, 166, 583, 330, 76, 686, 623, 942, 113, 690, 337, 22, 603, 123, 444, 830),
                    (576, 33, 522, 650, 778, 621, 143, 207, 79, 879, 851, 436, 701),
                    (320, 864, 872, 617, 395, 13, 847, 402, 883, 467, 822, 727, 344, 732, 159),
                    (448, 801, 129, 643, 391, 488, 846, 912, 18, 786, 629), (
                        770, 4, 135, 903, 780, 915, 153, 420, 165, 934, 561, 817, 323, 198, 722, 212, 863, 482, 227,
                        610,
                        486,
                        250, 252), (66, 933, 103, 907, 429, 239, 208, 431, 495, 117, 697, 604, 638),
                    (260, 138, 284, 163, 676, 554, 689, 945, 565, 311, 569, 699, 574, 836, 69, 966, 596, 89, 475, 348,
                     230),
                    (674, 805, 551, 635, 92, 882, 946, 947, 857, 505, 315, 956, 31),
                    (704, 705, 644, 648, 585, 298, 362, 620, 808, 111, 310, 56, 218, 446),
                    (897, 99, 388, 197, 998, 392, 170, 779, 300, 206, 719, 625, 757, 761, 157),
                    (736, 419, 104, 649, 266, 714, 652, 268, 180, 500, 125, 479),
                    (6, 200, 759, 236, 557, 219, 15, 684, 880, 20, 181, 54, 919, 661, 91, 540, 989),
                    (423, 904, 73, 489, 75, 490, 575, 845, 47, 875, 925, 914, 341, 118, 885, 26, 349, 255),
                    (773, 12, 531, 662, 920, 793, 154, 425, 570, 74, 725, 86, 343, 986, 233, 241, 116, 507, 895),
                    (512, 866, 390, 77, 974, 685, 464, 657, 717, 435, 564, 849, 980, 602),
                    (226, 357, 122, 105, 140, 622, 784, 537, 90, 380, 927),
                    (480, 640, 258, 642, 964, 261, 263, 556, 943, 469, 821, 665, 442, 826, 670),
                    (651, 795, 411, 931, 164, 172, 559, 304, 307, 441, 589, 336, 342, 867, 503, 743, 756, 887, 378),
                    (32, 929, 417, 833, 521, 937, 81, 145, 51, 721, 693, 755, 87, 351),
                    (160, 64, 96, 516, 421, 708, 584, 106, 365, 526, 941, 816, 209, 913, 27, 156, 957),
                    (192, 290, 483, 582, 199, 472, 874, 144, 176, 401, 926, 950, 23, 632, 922, 508, 542, 991),
                    (
                        0, 130, 900, 519, 911, 275, 660, 916, 790, 796, 669, 546, 567, 58, 193, 971, 248, 615, 240, 754,
                        504),
                    (902, 776, 404, 533, 667, 798, 161, 547, 293, 702, 457, 593, 608, 229, 363, 237, 750, 630, 631),
                    (898, 399, 272, 656, 852, 372, 598, 120, 730, 412, 671),
                    (396, 397, 788, 410, 45, 440, 573, 961, 707, 196, 710, 715, 462, 720, 861, 221, 355, 360, 244, 376),
                    (131, 68, 677, 680, 745, 108, 270, 335, 16, 19, 723, 443, 190, 639),
                    (896, 2, 17, 532, 534, 406, 959, 579, 204, 345, 999, 747, 109, 110, 369, 626, 628, 253, 126),
                    (289, 328, 809, 873, 43, 940, 271, 624, 474, 115, 789, 119, 282, 283, 668, 541, 414),
                    (449, 739, 740, 772, 167, 41, 11, 139, 203, 716, 687, 177, 211, 243, 381)}
    # Convert list of sets into dictionary
    dict_cluster = {c[0]: list(c[:]) for c in idx_clusters}
    return dict_cluster


#clusters = get_clusters()


# print(clusters)
# flag = is_incluster(195, 40, dict_cluster)

def gen_base(data, noise, clusters, cls_sizes=None):
    # Generate four classes: 0, 1, 2, 3, probably imbalanced
    if cls_sizes is None:
        cls_sizes = [10, 10, 10, 10]  # Set small variables for demo mode
    # Four-class classification, data generation
    MAX_MIX = 6  # Maximum number of mixtures signals
    MAX_AMP = 0.3  # V Minimum and maximum RMS amplitude of I/Q data signal
    MIN_AMP = 1.2  # V
    MAX_SHIFT = 7
    # Set a list of dict for the generated data
    db = []
    # For each class # Generate the sample set as a mixture of signals
    for cls, sample_size in enumerate(cls_sizes):
        for _ in range(sample_size):
            # For each new item in the sample set prepare a mixture
            # How many items are in the mixture?
            if cls < 3:  # 0: just noise, 1: single signal, 2: two signals
                cls_mix = cls
            else:
                cls_mix = cls # just make as many components is the mixture as the position number
                # TODO cls_mix = np.random.choice(list(range(3, MAX_MIX)))  # 3 or more signals

            # Each transmitter sends its unique code (no identical sources)
            idx_clus = np.random.choice(list(clusters.keys()), cls_mix, replace=False)
            idx_src = [np.random.choice(clusters.get(i), 1)[0] for i in idx_clus]

            coeffs = np.random.uniform(MIN_AMP, MAX_AMP, size=len(idx_src))
            signals = data[idx_src]
            signals = scale_complex(signals, coeffs)

            # mixture = noise[np.random.choice(noise.shape[0])]  # TODO restore: A mixture has its noise
            mixture = 0.01 * noise[np.random.choice(noise.shape[0])]  # A mixture has its noise
            mixture = scale_complex(mixture, [0.0001])
            mixture = mixture + np.sum(signals, axis=0)

            # Counting shifts from the basis vectors
            shifts = []
            for i, j in zip(idx_clus, idx_src):
                x = data[i]  # Cluster as basis
                y = data[j]  # To approximate
                err_min, best_b, best_shift = find_shiftX_exhaust(np.array([]), x, y, MAX_SHIFT)
                shifts.append(best_shift)
                x1 = shift_x(x, best_shift)
                # print(f'Class: {cls}, centroid: {i}, source: {j}, shift: {best_shift})')
            # Store all: mixture, its class, its sources, its coefficients
            db.append({
                "data": mixture,
                "label": cls,
                "source": idx_src,
                "basis": idx_clus,
                "coeff": coeffs,
                "shift": shifts
            })
            # print(f'Append class: {cls}, centroids: {idx_clus}, sources: {idx_src}, mixture coefficients: {coeffs}, shifts: {shfits}')
    return db


from scipy.linalg import lstsq


# IMPORTANT! Returns the Square of the 2-norm for (b - a x)"
#def scale_X(X, b): # Do you really neeed this one?
#  # Make a linear forecast
#  return X @ b

# Version of exhaustive shifting of the first vector of basis X of Feb 17th goes from 12_SingularValuesDecompositon
# ... and this file reads
# New variant of Feb 17 % IT IS ONLY IN THIS FILE .ipynb
def find_shiftX_exhaust(X, x, y, max_shift=11):
    # Two-parametric (scale, shift) self-modeling regression
    # Find the best phase of the scaled vector x, the first column of the matrix X
    # to minimize the euclidian distance err = |X @ b - y|.
    # max_shift is the maximum allowed phase shift in either direction
    # Works for complex-valued vectors
    # For the one-column X run this with X = np.column_stack([x])
    # Returns err, shift of X[:0] and scale of X
    err_min = float('inf')
    best_shift = 0
    len_x = len(x)

    # The shifting sub-function puts zeroes to the both ends of the signal
    def shift_it(x, len_x, shift):
        if shift < 0:
            shifted_x = np.pad(x[:len_x + shift], (abs(shift), 0), 'constant', constant_values=0)
        else:
            shifted_x = np.pad(x[shift:], (0, shift), 'constant', constant_values=0)
        return shifted_x

    # Scale x0 in X according to y
    def lsq_xy(X, x, y):
        if X.size == 0:
            Xx = np.column_stack([x])
        else:
            Xx = np.column_stack((X, x))
        b, err, rank, s = lstsq(Xx, y)
        if s is None:
            print('Warning: find_shiftX_exhaust -> lsq_xy -> lstsq has collinear vectors, err = 0.')
            raise ValueError('find_shiftX_exhaust -> lsq_xy -> lstsq has collinear vectors')
            err = 0  # FIXIT
        return err, b  # Distance and linear model parameters

    # Exhaustive search for +- max_shift
    for shift in range(- max_shift, max_shift + 1):
        shifted_x = shift_it(x, len_x, shift)
        err, b = lsq_xy(X, shifted_x, y)  # Current distance
        # print('Err', err, 'min', err_min, 'Shift', shift)
        if err < err_min:
            err_min = err
            best_b = b
            best_shift = shift
    return err_min, best_b, best_shift


# Version (exhaustive) of Wednedsay 12th, goes from 9_Distance_to_7bit
def find_lsq_shift_exhaust(x, y, N):
    # Semor, two-pametric (scale, shift) self-modeling regression
    # Find the best phase of the scaled y to minimize the distance to x
    # N is the maximum allowed phase shift in either direction
    # Works for complex-valued vectors
    # Returns original shifted y, shift of x and scale of x
    err_min = float('inf')
    best_shift = 0

    # The shifting subfunction puts zeroes to the both ends of the signal
    def shift_it(y, len_x, shift):
        if shift < 0:
            shifted_y = np.pad(y[:len_x + shift], (abs(shift), 0), 'constant', constant_values=0)
        else:
            shifted_y = np.pad(y[shift:], (0, shift), 'constant', constant_values=0)
        return shifted_y

    # Scale y according to x and return the distance
    def lsq_xy(x, y):
        b, err, rank, s = lstsq(np.column_stack([x]), y)
        return b[0], err  # Distance and scaled (and shifted)

    len_x = len(x)
    # Exhaustive search for +- N
    for shift in range(-N, N + 1):
        shifted_y = shift_it(y, len_x, shift)
        b0, err_0 = lsq_xy(x, shifted_y)  # Currest distance
        # print('Shift', shift * shift_dir, 'Err', err_0, 'min', err_min)
        if err_0 < err_min:
            err_min = err_0
            best_b0 = b0
            best_shift = shift
    return err_min, best_b0, -1 * best_shift


def shift_x(x, shift):
    # Shift a vector to several positions, replacing the gap with zeroes
    if shift < 0:
        x = np.pad(x[:len(x) + shift], (abs(shift), 0), 'constant', constant_values=0)
    else:
        x = np.pad(x[shift:], (0, shift), 'constant', constant_values=0)
    return x


def shift(X, shifts):
    # Call shift_x for each column in the matrix X
    for j in range(len(shifts)):
        X[:, j] = shift_x(X[:, j], shifts[j])
    return X


#------- Scale part --------------
def scale_complex_x(signal, coeff):
    # The signal is a complex vector to scale
    # Scale to max(abs) real and imag parts so that
    # The amplitude is the root mean square of I and Q the signal is coeff
    signal_rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    if signal_rms > 0:
        signal_scaled = signal * (coeff / signal_rms)
    else:
        signal_scaled = signal  # No scaling for zero
    return signal_scaled


def scale_complex(data, coeffs):
    # Each row of the data is a complex vector to scale
    # Scale to max(abs) real and imag parts so that
    # The amplitude is the root mean square of I and Q the signal is coeff.
    data_scaled = data.copy()
    for i, (signal, coeff) in enumerate(zip(data, coeffs)):
        signal_scaled = scale_complex_x(signal, coeff)
        data_scaled[i] = signal_scaled
    return data_scaled


def scale_separate_x(signal, coeff):
    # The signal is a complex vector to scale
    # Scale to max(abs) real and imag parts independently
    # Scale vecor by real coefficient
    max_real = np.max(np.abs(signal.real))
    if max_real == 0:
        signal_real = signal.real
    else:
        signal_real = signal.real / max_real
    max_imag = np.max(np.abs(signal.imag))
    if max_imag == 0:
        signal_imag = signal.imag
    else:
        signal_imag = signal.imag / np.max(np.abs(signal.imag))
    signal_scaled = coeff * signal_real + 1j * coeff * signal_imag
    return signal_scaled


def scale_separate(data, coeffs):
    # Each row of the data is a complex vector to scale
    # Scale to max(abs) real and imag parts independently
    # Scale each vecor by real coefficient
    data_scaled = data.copy()
    for i, (signal, coeff) in enumerate(zip(data, coeffs)):
        signal_scaled = scale_separate_x(signal, coeff)
        data_scaled[i] = signal_scaled
    return data_scaled


def is_incluster(key, qry, dict_cluster):  # Created ? is ever used ?
    if key in dict_cluster and qry in dict_cluster[key]:
        return True
    else:
        return False


#--------------------------------------------------------------------------------
# Version of Tuesday 11th, goes from 1_Simple_Regression
# from scipy.linalg import lstsq

def find_lsq_shift(x, y, N):  # Almost ready to ne obsoleted due to find_shiftX_exhaust
    # Semor, two-pametric (scale, shift) self-modeling regression
    # Find the best phase of the scaled y to minimize the distance to x
    # N is the maximum allowed phase shift in either direction
    # Works for complex-valued vectors
    # Returns original shifted y, shift of x and scale of x
    # min_dist = float('inf')
    best_shift = 0

    # The shifting subfunction puts zeroes to the both ends of the signal
    def shift_it(y, len_x, shift):
        if shift < 0:
            shifted_y = np.pad(y[:len_x + shift], (abs(shift), 0), 'constant', constant_values=0)
        else:
            shifted_y = np.pad(y[shift:], (0, shift), 'constant', constant_values=0)
        return shifted_y

    # Scale y according to x and return the distance
    def lsq_xy(x, y):
        b, err, rank, s = lstsq(np.column_stack([x]), y)
        return b[0], err  # Distance and scaled (and shifted)

    len_x = len(x)
    # Determine direction of decent
    b0_0, err_min = lsq_xy(x, y)
    b0_lft, err_lft = lsq_xy(x, shift_it(y, len_x, -1))
    b0_rgt, err_rgh = lsq_xy(x, shift_it(y, len_x, +1))

    if err_lft < err_min:
        best_b0 = b0_lft
        best_shift = shift_dir = -1
        err_min = err_lft  # Keep for the next step of descent
        if N == 1:
            return err_min, b0_lft, -1 * best_shift
    elif err_rgh < err_min:
        best_b0 = b0_rgt
        best_shift = shift_dir = +1
        err_min = err_rgh
        if N == 1:
            return err_min, b0_rgt, -1 * best_shift
    else:
        return err_min, b0_0, -1 * 0  # No shift is the best shift

    # Descent until N ends or min finds
    for shift in range(2, N + 1):
        shifted_y = shift_it(y, len_x, shift * shift_dir)
        b0, err_0 = lsq_xy(x, shifted_y)  # Currest distance
        # print('Shift', shift * shift_dir, 'Err', err_0, 'min', err_min)
        if err_0 < err_min:
            err_min = err_0  # Continue
            best_b0 = b0
            best_shift = shift
        else:
            break
    return err_min, best_b0, -1 * best_shift


# ** Scaling functions
def proj_xy(x, y):
    # Will be OBSOLETED soon
    # Computes the projection of vector x onto vector y
    # Works with dtype=complex
    if np.all(y == 0):
        raise ValueError("Projection is undefined for a zero vector y.")
    return (np.dot(x, y) / np.dot(y, y)) * y


def max_weighted_distance(x, y, kernel = None):
    # The two vectors are complex-valued
    if not kernel:
        # ker_Epanechnikov = [0.75, 0.9375, 1., 0.9375, 0.75]
        ker_Gaussian = [0.60, 0.77, 1., 0.77, 0.60]
        # ker_Alternative = [0.33, 0.60, 0.77, 1., 0.77, 0.60, 0.33]
        ker = ker_Gaussian
        # ker = ker_Alternative
        ker = ker / np.sum(ker)

    z = np.zeros(len(x))
    w2 = int(np.floor(len(ker)/2))
    for i in range(w2, len(x) - w2):
        # sum of  weighted squares
        z[i] = np.sum(np.abs(ker * (x[i-w2:i+w2+1] - y[i-w2:i+w2+1])**2))
    dist = max(z)
    return dist

def scale_x2max(x, amp=1 + 1j):
    # The expected amplitude of the iqdata is between 0.3 and 1.2 V. Set amp to 1 V
    # Works with dtype=complex
    if np.all(x == 0):
        raise ValueError("Scaling is impossible for a zero vector x.")
    z = np.zeros(len(x), dtype=complex)
    z.real = amp.real * x.real / np.max(np.abs(x.real))
    z.imag = amp.imag * x.imag / np.max(np.abs(x.imag))
    return z


def complex_abs(x, y):  # Ready to be obsoleted?
    # Absolute values of the residues elements of two complex vectors
    # Works with dtype=complex
    # TODO needs additional cheking, do we need this function
    z = np.empty(x.shape, dtype=complex)
    z.real = np.abs(x.real - y.real)
    z.imag = np.abs(x.imag - y.imag)
    return z


# ** Distance functions
def dist_xy(x, y):
    # Works with dtype=complex
    # TODO remove this function due to it is obvious
    return np.linalg.norm(x - y)


def check_consecutive_sum(x, N, C):
    # Checks if the sum of any N consecutive elements in the vector does not exceed C
    # Works with dtype=complex
    if N > len(x):
        return np.sum(x.real) <= C and np.sum(x.imag) <= C  # If N is larger than the vector length, check entire sum
    current_sum = sum(x[:N])  # Compute initial window sum
    if current_sum.real > C or current_sum.imag > C:
        return False
    for i in range(N, len(x)):  # Use sliding window
        current_sum += x[i] - x[i - N]
        if current_sum.real > C or current_sum.imag > C:
            return False
    return True


# ** Semor, two-pametric (scale, shift) self-modeling regression
def find_best_shift(x, y, N, is_proj=False):
    # Find the best phase of y to minimize the distance to x
    # N is the maximum allowed phase shift in either direction
    # Works for complex-valued vectors
    min_residue = float('inf')
    best_shift = 0

    # The shifting runs two times, put it to a function
    def shift_it(x, y, len_x, shift):
        if shift < 0:
            shifted_y = np.pad(y[:len_x + shift], (abs(shift), 0), 'constant', constant_values=0)
        else:
            shifted_y = np.pad(y[shift:], (0, shift), 'constant', constant_values=0)
        return shifted_y

    #--- end subfunction

    len_x = len(x)
    for shift in range(-N, N + 1):
        shifted_y = shift_it(x, y, len_x, shift)

        if is_proj:
            shifted_y = proj_xy(x, shifted_y)
        residue = np.linalg.norm(x - shifted_y)

        if residue < min_residue:
            min_residue = residue
            best_shift = shift

    shifted_y = shift_it(x, y, len_x, best_shift)
    if is_proj:
        shifted_y = proj_xy(x, shifted_y)
    return shifted_y  # [best_shift, scale] as the model parameters

    # TODO def collect_cluster(X, idx_centroid, idx_general, CRAD, WLEN): (from Colab 9_)
    # TODO def find_centroid(cluster): (from Colab 9_)
