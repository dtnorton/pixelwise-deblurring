function main()
    % Parse arguments
    args = parse_args();

    % Load .mat file
    data = load(args.matfile);
    timestamps = data.t(1, :) - data.t(1, 1); % Creating a local timescale
    obs_frames = data.frames;
    num_frames = length(timestamps);

    N = args.N;
    if ~isequal(log2(N - 1), floor(log2(N - 1)))
        warning('Window size (%d) is not of the form 2^l + 1. Unreliable results.', N);
    end

    parfor i = 1:length(args.indices)
        idx = args.indices(i);
        fprintf('Processing Frames: [%d, %d]\n', idx, idx + N);

        % Verify enough frames are available
        if idx + N > num_frames
            fprintf('Skipping frame %d since the matfile does not have the subsequent N frames\n', idx);
            continue;
        end

        obs_t = timestamps(idx:idx + N - 1);
        obs_frames_subset = obs_frames(idx:idx + N - 1, :, :);
        [img_width, img_height] = size(obs_frames_subset(:, :, 1));
        y_reshaped = reshape(obs_frames_subset, [N, img_width * img_height]);
        y_mean = mean(y_reshaped, 1);

        % Check timestamps are uniformly spaced
        t_samples = linspace(obs_t(1), obs_t(1) + (N - 1) * args.T, N);
        if norm(t_samples - obs_t) > args.T
            warning('Observation timestamps do not seem to be uniformly spaced. Approximation by uniform spacing will affect results.');
        end

        [V, H] = construct_problem_matrices(t_samples, args.tau, args.n);
        Z = full(V * H');

        pixels = cell(1, size(y_reshaped, 2));
        for j = 1:size(y_reshaped, 2)
            Y = construct_Y(t_samples, y_reshaped(:, j) - y_mean(j), args.tau);
            pixels{j} = {Y, j};
        end

        timescale = linspace(t_samples(1), t_samples(end), 2^args.n);
        x_est = zeros(length(timescale), size(y_reshaped, 2));

        tic;
        parfor j = 1:length(pixels)
            Yj = pixels{j};
            results = optim_wrapper(Yj{1}, @perform_lasso, Z, args.lmd, args.kappa);
            D_est_j = results{1};
            x_est_j = y_mean(Yj{2}) + H' * D_est_j;
            x_est(:, Yj{2}) = x_est_j;
        end
        toc;

        est_frames_subset = reshape(x_est, [length(timescale), img_width, img_height]);
        save(sprintf('%s_%06d_%06d.mat', args.output_prefix, idx, idx + N), 'obs_frames_subset', 'est_frames_subset', 't_samples', 'timescale');
    end
end

function args = parse_args()
    p = inputParser;
    addParameter(p, 'tau', 0.011, @isnumeric);
    addParameter(p, 'n', 8, @isnumeric);
    addParameter(p, 'N', 33, @isnumeric);
    addParameter(p, 'T', 0.005, @isnumeric);
    addParameter(p, 'lmd', 0.1, @isnumeric);
    addParameter(p, 'kappa', 0.1, @isnumeric);
    addParameter(p, 'matfile', '', @ischar);
    addParameter(p, 'indices', [], @isnumeric);
    addParameter(p, 'output_prefix', '', @ischar);

    parse(p);
    args = p.Results;
end
