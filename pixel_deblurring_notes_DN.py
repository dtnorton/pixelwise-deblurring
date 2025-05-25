import numpy as np
import textwrap
import time

from cvxopt import matrix, solvers
from scipy import sparse


def process_stack(obs_frames, timestamps, N=33, tau=0.011, n=8, T=0.005,
                  lmd=0.1, kappa=0.1):

    nframes, nrows, ncols = obs_frames.shape

    # Ignore threading for now...
    for iframe in np.arange(0, nframes, N):
        obs_t = timestamps[iframe:iframe + N]
        frames = obs_frames[iframe:iframe + N]

        y_reshaped = frames.reshape((N, -1))
        y_mean = np.mean(y_reshaped, axis=0)

        t_samples = np.linspace(obs_t[0], obs_t[0] + (N - 1) * T, num=N)
        time_offset_flag = False
        if np.linalg.norm(t_samples - obs_t) > T:
            time_offset_flag = True
            # message(msg)

        V, H = construct_problem_matrices(t_samples, tau, n)
        H = H.transpose()
        # TODO: I don't think this requires a .toarray() method call.
        # Z = V.dot(H).toarray()
        Z = V.dot(H)

        pixels = []
        for ipixel in range(y_reshaped.shape[1]):
            Y = construct_Y(
                t_samples, y_reshaped[:, ipixel] - y_mean[ipixel], tau)
            pixels.append((Y, ipixel))

        timescale = np.linspace(t_samples[0], t_samples[-1], 2**n,
                                endpoint=False)
        nsamp = timescale.size
        x_est = np.zeros((timescale.size, y_reshaped.shape[1]))

        # for ipixel, results in NEED_TO_FIGURE_THIS_OUT:
        #     D_est_ipixel = results[0]
        #     x_est_ipixel = y_mean[ipixel] + H.dot(D_est_ipixel)
        #     x_est[:, ipixel] = x_est_ipixel

        # est_frames_subset = x_est.reshape((nsamp, nrows, ncols))

        for ipixel in np.arange(nsamp):
            Ypix = pixels[ipixel][0]
            D_est_ipixel, _ = perform_lasso(Ypix, Z, lmd, kappa)


def message(msg):
    msg = textwrap.fill(msg, 78)
    print(msg)


def perform_lasso(Y, Z, lmd, kappa):
    # Suppress solver output
    # solvers.options['show_progress'] = False

    # Dimension of the coefficient vector
    K = Z.shape[1]
    Q_ = 2 * kappa * Z.T.dot(Z)
    # Q = np.concatenate([np.concatenate([Q_, -Q_]), np.concatenate([-Q_, Q_])], axis=1)
    Q = sparse.vstack([sparse.vstack([Q_, -Q_]), sparse.vstack([-Q_, Q_])])
    e = np.ones(K)
    c = np.concatenate([-2 * kappa * Z.T.dot(Y) + lmd * e,
                       2 * kappa * Z.T.dot(Y) + lmd * e])  # objective function

    # Convert numpy arrays to cvxopt matrices
    Q = matrix(Q.todense())
    c = matrix(c)

    try:
        tic = time.time()
        sol = solvers.qp(Q, c)
        toc = time.time()

        qp_sol = np.array(sol['x']).flatten()
        # shrink components less than 1e-3*max(qp_sol)
        qp_sol[qp_sol < 1e-3 * np.max(qp_sol)] = 0

        D_pos = qp_sol[:K]
        D_neg = qp_sol[K:]
        D_star = D_pos - D_neg
    except Exception as exc:
        D_star = np.zeros(K)
        print(exc)
        return D_star, {'compute_time': np.inf, 'Q': Q, 'c': c,
                        'qp_sol': None}

    return D_star, {'compute_time': toc-tic, 'Q': Q, 'c': c,
                    'qp_sol': qp_sol, 'status': sol['status']}


def optim_wrapper(Yj, func, Z, lmd, kappa):
    return Yj[1], func(Yj[0], Z, lmd, kappa)


def _construct_V(t_obs, tau, K):
    N = t_obs.shape[0]
    total_time = t_obs[-1] - t_obs[0]

    M = N-1
    m = 0
    row = []
    col = []
    data = []
    for i in range(1, N):
        # add y(t_i), y(t_0) constraint
        ti_t0 = t_obs[i] - t_obs[0]
        outer_term_exponent = ti_t0 / tau
        for k in range(K):
            lower_limit = np.maximum(k*total_time/(tau*K), 0)
            upper_limit = np.minimum((k+1)*total_time/(tau*K), ti_t0/tau)
            if lower_limit < upper_limit:
                val = np.exp(upper_limit-outer_term_exponent) - \
                    np.exp(lower_limit-outer_term_exponent)
                row.append(m)
                col.append(k)
                data.append(val)
        m += 1
    assert m == M, "Number of constraints seems to be wrongly calculated"
    V = sparse.coo_matrix((data, (row, col)), shape=(M, K), dtype=np.float64)
    return V.tocsr()


def _construct_H(haar_level):
    # Size of H matrix
    K = np.power(2, haar_level)

    row = []
    col = []
    data = []

    # scaling function phi_00
    for k in range(K):
        row.append(0)
        col.append(k)
        data.append(1)

    # haar wavelets
    for l in range(1, K):
        n = np.floor(np.log2(l))
        k = l - 2**n
        # this row corresponds to \psi_{n,k} which has support in k/2**n, (k+1)/2**n with half being positive and
        # other half being negative
        # NOTE: int is used instead of floor since the indices are all positive numbers
        idx_min = int(K*k/2**n)
        idx_mid = int(K*(k+0.5)/2**n)
        idx_max = int(K*(k+1)/2**n)
        for m in range(idx_min, idx_mid):
            row.append(l)
            col.append(m)
            data.append(2**(n/2))
        for m in range(idx_mid, idx_max):
            row.append(l)
            col.append(m)
            data.append(-2**(n/2))
    H = sparse.coo_matrix((data, (row, col)), shape=(
        K, K), dtype=np.float64) / np.sqrt(2**haar_level)
    return H.tocsr()


def construct_Y(t_obs, y_obs, tau):
    #  y_obs are measurements in Kelvin

    assert len(y_obs.shape) == 1, "Each pixel should be independently solved"
    N = y_obs.shape[0]

    Y = []
    for i in range(1, N):
        # add y(t_i), y(t_0) constraint
        Y.append(y_obs[i] - y_obs[0] * np.exp(-(t_obs[i] - t_obs[0]) / tau))
    return np.array(Y)


def construct_problem_matrices(t_obs, tau, haar_level):
    # Util function to construct V and H given tau, haar_level and [t_0, t_1, ... , t_N]
    # t_obs are timestamps of observations in seconds
    # tau is the thermal time constant of the microbolometer in seconds
    # haar_level determines time resolution of the estimate i.e [t_0, t_N) is split into 2**haar_level bins
    # returns: V, H

    assert len(
        t_obs.shape) == 1 and t_obs.shape[0] >= 2, "At least two observations are required"

    V = _construct_V(t_obs, tau, 2**haar_level)
    H = _construct_H(haar_level)

    return V, H
