# * Collection of functions for I/D data signal processing
# The signal dtype=complex
import numpy as np


from scipy.linalg import lstsq

# IMPORTANT! Returns the Square of the 2-norm for (b - a x)"
#def scale_X(X, b): # Do you really need this one?
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
    # For two time series define a metric distance function so that be stable to synchronous
    # variations of the time series in time, but emphasises asynchronous ones.
    # Use a modification of the Hausdorff metric. Slide both time series with kernel window and
    # # find the maximum weighted difference.
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
    # Note: needs additional cheking, do we need this function
    z = np.empty(x.shape, dtype=complex)
    z.real = np.abs(x.real - y.real)
    z.imag = np.abs(x.imag - y.imag)
    return z


# ** Distance functions
def dist_xy(x, y):
    # Works with dtype=complex
    # Note: remove this function due to it is obvious
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
