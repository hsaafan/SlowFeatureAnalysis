import numpy as np
import scipy.linalg as la
from scipy import stats
import random
import matplotlib.pyplot as plt


def CDC_index(sqrt_M: np.ndarray, x: np.ndarray, xi_i: np.ndarray):
    return((xi_i.T @ sqrt_M @ x) ** 2)


def PDC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray):
    return(x.T @ M @ xi_i @ xi_i.T @ x)


def DC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray):
    return(x.T @ xi_i @ xi_i.T @ M @ xi_i @ xi_i.T @ x)


def RBC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray):
    return(xi_i.T @ M @ x) ** 2 / (xi_i.T @ M @ xi_i)


def rCDC_index(sqrt_M: np.ndarray, x: np.ndarray, xi_i: np.ndarray,
               S: np.ndarray):
    CDC = CDC_index(sqrt_M, x, xi_i)
    return(CDC / (xi_i.T @ S @ sqrt_M @ sqrt_M @ xi_i))


def rPDC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray, S: np.ndarray):
    PDC = PDC_index(M, x, xi_i)
    return(PDC / (xi_i.T @ S @ M @ xi_i))


def rDC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray, S: np.ndarray):
    return((x.T @ xi_i @ xi_i.T @ x) / (xi_i.T @ S @ xi_i))


def rRBC_index(M: np.ndarray, x: np.ndarray, xi_i: np.ndarray, S: np.ndarray):
    return((xi_i.T @ M @ x) ** 2 / (xi_i.T @ M @ S @ M @ xi_i))


def CDC_limit(M: np.ndarray, xi_i: np.ndarray, S: np.ndarray, alpha: float):
    static = xi_i.T @ S @ M @ xi_i
    upper = stats.chi2.ppf(1 - alpha, 1) * static
    lower = stats.chi2.ppf(alpha, 1) * static
    return(lower, upper)


def PDC_limit(M: np.ndarray, xi_i: np.ndarray, S: np.ndarray, alpha: float):
    stdv = ((xi_i.T @ S @ M @ xi_i) ** 2
            + xi_i.T @ S @ M @ M @ xi_i @ xi_i.T @ S @ xi_i) ** 0.5
    static = xi_i.T @ S @ M @ xi_i
    upper = static + 3 * stdv
    lower = static - 3 * stdv
    return(lower, upper)


def DC_limit(M: np.ndarray, xi_i: np.ndarray, S: np.ndarray, alpha: float):
    static = xi_i.T @ S @ xi_i @ xi_i.T @ M @ xi_i
    upper = stats.chi2.ppf(1 - alpha, 1) * static
    lower = stats.chi2.ppf(alpha, 1) * static
    return(lower, upper)


def RBC_limit(M: np.ndarray, xi_i: np.ndarray, S: np.ndarray, alpha: float):
    static = xi_i.T @ M @ S @ M @ xi_i / (xi_i.T @ M @ xi_i)
    upper = stats.chi2.ppf(1 - alpha, 1) * static
    lower = stats.chi2.ppf(alpha, 1) * static
    return(lower, upper)


INDEX_FUNCTIONS = {
    'CDC': CDC_index,
    'PDC': PDC_index,
    'DC': DC_index,
    'RBC': RBC_index
}

RELATIVE_INDEX_FUNCTIONS = {
    'rCDC': rCDC_index,
    'rPDC': rPDC_index,
    'rDC': rDC_index,
    'rRBC': rRBC_index
}

LIMIT_FUCNTIONS = {
    'CDC': CDC_limit,
    'PDC': PDC_limit,
    'DC': DC_limit,
    'RBC': RBC_limit
}


def contribution_index(M: np.ndarray, x: np.ndarray, indices: list = ['CDC']):

    indices = check_indices_are_valid(indices, INDEX_FUNCTIONS.keys())
    M, x = check_index_inputs(M, x)
    n = M.shape[0]
    index_values = dict()

    if 'CDC' in indices:
        sqrt_M = la.fractional_matrix_power(M, 0.5)

    for ind in indices:
        vals = [0] * n
        for i in range(n):
            xi_i = xi(n, i)
            if not ind == 'CDC':
                vals[i] = INDEX_FUNCTIONS[ind](M, x, xi_i)
            else:
                vals[i] = INDEX_FUNCTIONS[ind](sqrt_M, x, xi_i)
        index_values[ind] = vals
    return(index_values)


def relative_contribution_index(M: np.ndarray, x: np.ndarray, S: np.ndarray,
                                indices: list = ['rCDC']):

    indices = check_indices_are_valid(indices, RELATIVE_INDEX_FUNCTIONS.keys())
    M, x = check_index_inputs(M, x)
    M, S = check_control_limit_inputs(M, S)
    n = M.shape[0]
    index_values = dict()

    if 'rCDC' in indices:
        sqrt_M = la.fractional_matrix_power(M, 0.5)

    for ind in indices:
        vals = [0] * n
        for i in range(n):
            xi_i = xi(n, i)
            if not ind == 'rCDC':
                vals[i] = RELATIVE_INDEX_FUNCTIONS[ind](M, x, xi_i, S)
            else:
                vals[i] = RELATIVE_INDEX_FUNCTIONS[ind](sqrt_M, x, xi_i, S)
        index_values[ind] = vals
    return(index_values)


def contribution_control_limits(M: np.ndarray, S: np.ndarray, alpha: float,
                                indices: list = ['rCDC']):

    indices = check_indices_are_valid(indices, INDEX_FUNCTIONS.keys())
    M, S = check_control_limit_inputs(M, S)
    n = M.shape[0]
    index_values = dict()

    for ind in indices:
        upper = [0] * n
        lower = [0] * n
        for i in range(n):
            xi_i = xi(n, i)
            lower[i], upper[i] = LIMIT_FUCNTIONS[ind](M, xi_i, S, alpha)
        index_values[ind] = (lower, upper)
    return(index_values)


def xi(n: int, i: int):
    xi = np.zeros((n, 1))
    xi[i] = 1
    return(xi)


def check_indices_are_valid(indices: list, valid_indices: list):
    if type(indices) == str:
        indices = [indices]

    for ind in indices:
        if ind not in valid_indices:
            raise ValueError(f"No contribution index {ind} exists")
    return(indices)


def check_index_inputs(M, x):
    # Input checks
    if not (isinstance(M, np.ndarray) and isinstance(x, np.ndarray)):
        raise TypeError("Expected numpy array inputs for M and x")
    if not (M.shape[0] == M.shape[1] == x.size):
        raise ValueError("M needs to be an [n x n] matrix and x needs to be "
                         "an [n x 1] vector")
    x = np.reshape(x, (-1, 1))  # Makes sure it's a column vector
    return(M, x)


def check_control_limit_inputs(M, S):
    # Input checks
    if not (isinstance(M, np.ndarray) and isinstance(S, np.ndarray)):
        raise TypeError("Expected numpy array inputs for M and S")
    if not (M.shape[0] == M.shape[1] == S.shape[0] == S.shape[1]):
        raise ValueError("M needs to be an [n x n] matrix and S needs to be "
                         "an [n x n] matrix")
    return(M, S)


def example_process_model(num_samples):
    A = [
        [-0.3441, 0.4815, 0.6637],
        [-0.2313, -0.5936, 0.3545],
        [-0.5060, 0.2495, 0.0739],
        [-0.5552, -0.2405, -0.1123],
        [-0.3371, 0.3822, -0.6115],
        [-0.3877, -0.3868, -0.2045]
    ]
    A = np.asarray(A)
    num_vars = 6

    # Generate inputs t
    t1 = 2.0 * stats.uniform.rvs(size=num_samples)
    t2 = 1.6 * stats.uniform.rvs(size=num_samples)
    t3 = 1.2 * stats.uniform.rvs(size=num_samples)
    t = np.asarray([t1, t2, t3])

    # Generate noise
    noise = [None] * num_vars
    for i in range(num_vars):
        noise[i] = stats.norm.rvs(size=num_samples, scale=0.2)
    noise = np.asarray(noise)

    # Create samples
    X = A @ t + noise

    return(X)


if __name__ == "__main__":
    num_samples = 3000
    num_faults = 2000
    num_vars = 6

    X = example_process_model(num_samples)

    """ PCA Model """
    # Shift to 0 mean
    xmean = np.mean(X, 1).reshape((-1, 1))
    X = X - xmean

    # Scale to unit variance
    xstd = np.std(X, 1).reshape((-1, 1))
    X = X / xstd

    assert np.allclose(np.mean(X, 1), 0)
    assert np.allclose(np.std(X, 1), 1)

    S = np.cov(X)
    Lam, P = la.eig(S)
    order = np.argsort(-1 * Lam)
    Lam = Lam[order]
    P = P[:, order]

    # Plot cumulative variance of eigenvectors
    # cum_eig = np.cumsum(Lam) / np.sum(Lam)
    # plt.plot(cum_eig)
    # plt.show()
    principal_vectors = 3
    alpha = 0.01  # Confidence = (1 - alpha) x 100%

    P_resid = P[:, principal_vectors:]
    Lam_resid = Lam[principal_vectors:]
    P = P[:, :principal_vectors]
    Lam = Lam[:principal_vectors]
    D = P @ np.diag(Lam ** -1) @ P.T

    # Generate faults
    faults = np.zeros((num_vars, num_faults))

    for fault_sample in range(num_faults):
        fault_var = random.sample(range(num_vars), 1)[0]
        faults[fault_var, fault_sample] = 5.0 * stats.uniform.rvs()

    X_faulty = example_process_model(num_faults) + faults
    X_faulty = (X_faulty - xmean) / xstd

    T_sqr = [0] * num_faults
    for i in range(num_faults):
        T_sqr[i] = X_faulty[:, i].T @ D @ X_faulty[:, i]

    T_sqr_limit = [stats.chi2.ppf(1 - alpha, principal_vectors)] * num_faults

    detected_faults = []

    for i in range(num_faults):
        if T_sqr[i] > T_sqr_limit[i]:
            detected_faults.append(i)

    fault_detect_rate = len(detected_faults) / num_faults * 100
    print(f"T^2 Detected Faults: {fault_detect_rate:.2f} %")
    # plt.plot(T_sqr, label="\$T^2\$")
    # plt.plot(T_sqr_limit, label="Limit")
    # plt.legend()
    # plt.show()
    all_indices = list(INDEX_FUNCTIONS.keys())
    cont_rates = dict()
    for ind in all_indices:
        # Tracks number of correct diagnoses, false diagnoses, and missed
        # diagnoses
        cont_rates[ind] = [0, 0, 0]

    limits = contribution_control_limits(D, S, alpha, all_indices)
    for i in detected_faults:
        # Get index and limit for each fault sample
        cont = contribution_index(D, X_faulty[:, i], all_indices)

        for ind in all_indices:
            for j in range(num_vars):
                # Correct if index above control limit and the variable being
                # tested is the right direction for the fault
                if limits[ind][0][j] < cont[ind][j] < limits[ind][1][j]:
                    if j == np.argmax(faults[:, i]):
                        cont_rates[ind][0] += 1
                    else:
                        cont_rates[ind][1] += 1
                elif j == np.argmax(faults[:, i]):
                    cont_rates[ind][2] += 1

    for ind in all_indices:
        diag_rate = cont_rates[ind][0] / len(detected_faults) * 100
        false_diag_rate = cont_rates[ind][1] / len(detected_faults) * 100
        missed_rate = cont_rates[ind][2] / len(detected_faults) * 100
        print(f"\n{ind} correct diagnosis: {diag_rate:.2f} %")
        print(f"{ind} false diagnosis: {false_diag_rate:.2f} %")
        print(f"{ind} missed diagnosis: {missed_rate:.2f} %")
