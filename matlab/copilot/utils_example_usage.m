% Example usage
tau = 0.011;
haar_level = 8;
t_obs = linspace(0, 1, 33);

[V, H] = construct_problem_matrices(t_obs, tau, haar_level);
y_obs = rand(size(t_obs));
Y = construct_Y(t_obs, y_obs, tau);
