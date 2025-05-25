function [D_star, results] = perform_lasso(Y, Z, lmd, kappa)
    K = size(Z, 2);
    Q_ = 2 * kappa * (Z' * Z);
    Q = [Q_, -Q_; -Q_, Q_];
    e = ones(K, 1);
    c = [-2 * kappa * (Z' * Y) + lmd * e; 2 * kappa * (Z' * Y) + lmd * e];
    
    % Define the bounds and initial conditions
    lb = zeros(2 * K, 1);  % Non-negative constraints
    ub = [];

    try
        options = optimoptions('quadprog', 'Display', 'off');
        tic;
        qp_sol = quadprog(Q, c, [], [], [], [], lb, ub, [], options);
        toc;

        % Shrink components less than 1e-3*max(qp_sol)
        qp_sol(qp_sol < 1e-3 * max(qp_sol)) = 0;

        D_pos = qp_sol(1:K);
        D_neg = qp_sol(K + 1:end);
        D_star = D_pos - D_neg;
        results = struct('compute_time', toc, 'Q', Q, 'c', c, 'qp_sol', qp_sol);
    catch exception
        D_star = zeros(K, 1);
        results = struct('compute_time', Inf, 'Q', Q, 'c', c, 'qp_sol', []);
        disp(exception.message);
    end
end

function [index, D_star] = optim_wrapper(Yj, func, Z, lmd, kappa)
    index = Yj{2};
    D_star = func(Yj{1}, Z, lmd, kappa);
end

function pixel_wise_deblurring(tau, n, N, T, lmd, kappa, matfile, indices, output_prefix)
    % Default argument values
    if nargin < 1, tau = 0.011; end
    if nargin < 2, n = 8; end
    if nargin < 3, N = 33; end
    if nargin < 4, T = 0.005; end
    if nargin < 5, lmd = 0.1; end
    if nargin < 6, kappa = 0.1; end
    if nargin < 7, error('matfile is required'); end
    if nargin < 8, error('indices are required'); end
    if nargin < 9, error('output_prefix is required'); end

    % Reading the .mat file
    data = load(matfile);
    timestamps = data.t(1, :) - data.t(1, 1);  % Creating a local timescale
    obs_frames = data.frames;
    num_frames = length(timestamps);

    if log2(N - 1) ~= fix(log2(N - 1))
        warning('Window size (%d) is not of the form 2**l + 1. Unreliable results.', N);
    end

    % Parallel pool
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool;
    end

    % Process each frame index
    parfor i = indices
        fprintf('Processing Frames: [%d, %d]\n', i, i + N);

        if i + N > num_frames
            fprintf('Skipping frame %d since the matfile doesn''t have the subsequent N frames\n', i);
            continue;
        end

        obs_t = timestamps(i:i + N - 1);
        obs_frames_subset = obs_frames(i:i + N - 1, :, :);
        [img_width, img_height, ~] = size(obs_frames_subset);
        y_reshaped = reshape(obs_frames_subset, [N, img_width * img_height]);
        y_mean = mean(y_reshaped, 1);

        % Ensure timestamps are uniformly spaced
        t_samples = linspace(obs_t(1), obs_t(1) + (N - 1) * T, N);
        if norm(t_samples - obs_t) > T
            warning('Observation timestamps don''t seem to be uniformly spaced. Approximation by uniform spacing will affect results');
        end

        [V, H] = construct_problem_matrices(t_samples, tau, n);
        Z = V * H';

        pixels = cell(1, size(y_reshaped, 2));
        for j = 1:size(y_reshaped, 2)
            Y = construct_Y(t_samples, y_reshaped(:, j) - y_mean(j), tau);
            pixels{j} = {Y, j};
        end

        timescale = linspace(t_samples(1), t_samples(end), 2^n);
        x_est = zeros(length(timescale), size(y_reshaped, 2));

        tic;
        parfor j = 1:length(pixels)
            [~, D_star] = optim_wrapper(pixels{j}, @perform_lasso, Z, lmd, kappa);
            x_est_j = y_mean(pixels{j}{2}) + H' * D_star;
            x_est(:, pixels{j}{2}) = x_est_j;
        end
        toc;

        est_frames_subset = reshape(x_est, [length(timescale), img_width, img_height]);
        save(sprintf('%s_%06d_%06d.mat', output_prefix, i, i + N), 'obs_frames_subset', 'est_frames_subset', 't_samples', 'timescale');
    end
end
