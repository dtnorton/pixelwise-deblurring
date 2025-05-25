function [D_star, metadata] = perform_lasso(Y, Z, lmd, kappa)
    try
        % Dimension of the coefficient vector
        K = size(Z, 2);

        % Compute Q and c
        Q_ = 2 * kappa * (Z' * Z);
        Q = [Q_, -Q_; -Q_, Q_];
        e = ones(K, 1);
        c = [-2 * kappa * (Z' * Y) + lmd * e; 2 * kappa * (Z' * Y) + lmd * e];

        % Define lower and upper bounds for variables (no constraints => -Inf to Inf)
        lb = zeros(2 * K, 1);
        ub = [];

        % Solve the quadratic program using quadprog
        options = optimoptions('quadprog', 'Display', 'none');
        tic;
        [qp_sol, ~, exitflag] = quadprog(Q, c, [], [], [], [], lb, ub, [], options);
        compute_time = toc;

        if exitflag ~= 1
            error('Optimization did not converge');
        end

        % Shrink components less than 1e-3 * max(qp_sol)
        qp_sol(qp_sol < 1e-3 * max(qp_sol)) = 0;

        D_pos = qp_sol(1:K);
        D_neg = qp_sol(K+1:end);
        D_star = D_pos - D_neg;

        metadata = struct('compute_time', compute_time, 'Q', Q, 'c', c, 'qp_sol', qp_sol, 'status', exitflag);
    catch ME
        % In case of error, return zero coefficients and metadata
        D_star = zeros(K, 1);
        metadata = struct('compute_time', Inf, 'Q', Q, 'c', c, 'qp_sol', [], 'status', -1, 'error_message', ME.message);
    end
end

function result = optim_wrapper(Yj, func, Z, lmd, kappa)
    % Wrapper function
    result = {Yj(2), func(Yj(1), Z, lmd, kappa)};
end
