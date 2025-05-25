function V = construct_V(t_obs, tau, K)
    N = length(t_obs);
    total_time = t_obs(end) - t_obs(1);

    M = N - 1;
    m = 0;
    row = [];
    col = [];
    data = [];

    for i = 2:N
        % Add y(t_i), y(t_0) constraint
        ti_t0 = t_obs(i) - t_obs(1);
        outer_term_exponent = ti_t0 / tau;
        for k = 0:(K-1)
            lower_limit = max(k * total_time / (tau * K), 0);
            upper_limit = min((k + 1) * total_time / (tau * K), ti_t0 / tau);
            if lower_limit < upper_limit
                val = exp(upper_limit - outer_term_exponent) - exp(lower_limit - outer_term_exponent);
                row = [row; m];
                col = [col; k];
                data = [data; val];
            end
        end
        m = m + 1;
    end

    assert(m == M, 'Number of constraints seems to be wrongly calculated');

    V = sparse(row, col, data, M, K);
end

function H = construct_H(haar_level)
    K = 2^haar_level;

    row = [];
    col = [];
    data = [];

    % Scaling function phi_00
    for k = 1:K
        row = [row; 1];
        col = [col; k];
        data = [data; 1];
    end

    % Haar wavelets
    for l = 2:K
        n = floor(log2(l-1));
        k = l - 1 - 2^n;

        idx_min = floor(K * k / 2^n) + 1;
        idx_mid = floor(K * (k + 0.5) / 2^n) + 1;
        idx_max = floor(K * (k + 1) / 2^n) + 1;

        for m = idx_min:(idx_mid - 1)
            row = [row; l];
            col = [col; m];
            data = [data; 2^(n / 2)];
        end
        for m = idx_mid:(idx_max - 1)
            row = [row; l];
            col = [col; m];
            data = [data; -2^(n / 2)];
        end
    end

    H = sparse(row, col, data, K, K) / sqrt(2^haar_level);
end

function Y = construct_Y(t_obs, y_obs, tau)
    assert(ndims(y_obs) == 2 && size(y_obs, 2) == 1, 'Each pixel should be independently solved');

    N = length(y_obs);
    Y = [];
    for i = 2:N
        % Add y(t_i), y(t_0) constraint
        Y = [Y; y_obs(i) - y_obs(1) * exp(-(t_obs(i) - t_obs(1)) / tau)];
    end
end

function [V, H] = construct_problem_matrices(t_obs, tau, haar_level)
    assert(isvector(t_obs) && length(t_obs) >= 2, 'At least two observations are required');

    V = construct_V(t_obs, tau, 2^haar_level);
    H = construct_H(haar_level);
end
