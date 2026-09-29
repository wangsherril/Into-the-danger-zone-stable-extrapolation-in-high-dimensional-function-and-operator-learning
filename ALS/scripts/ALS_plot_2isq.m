

clear all; close all; clc;
addpath(genpath('../utils'))

%% Define main parameters %%
% parpool('local');
[ms, lw, fs, colors, markers, AlphaLevel] = get_fig_param();
poly_type = 'legendre';
% fun_name = 'aniso_exp';
fun_name = 'nonintegrable_singularity_delta_2ia';
% fun_name = 'reciprocal_linear';
% d = 32;
% OOD_type = "unbounded"; % hypercube or unbounded

% Automatically run all four setups:
% d = 32, unbounded
% d = 32, hypercube
% d = 8,  unbounded
% d = 8,  hypercube
setups = {
    32, "unbounded";
    32, "hypercube";
    8, "unbounded";
    8, "hypercube";
};

space = ' ';
func = str2func(fun_name); % convert function name to function handle
beta = 0.5; % used in the bulk procedure

m_max = 1.5e+4;
scale_type = 'log';
num_trials = 30;
a_val_vec = [2];
% b_val_vec = [1, 1.1, 1.2, 1.3, 1.6, 1.8];  % hypercube
% b_val_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];  %unbounded    

K = 10000; % error grid size


for setup_idx = 1:size(setups,1)
    d = setups{setup_idx,1};
    OOD_type = setups{setup_idx,2};

    if OOD_type == "unbounded"
        b_val_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];

    elseif OOD_type == "hypercube"
        b_val_vec = [1, 1.1, 1.2, 1.3, 1.6, 1.8];
    end

    if d == 8
        c_scale = 3;
    elseif d ==32
        c_scale = 1;
    end

    if isequal(scale_type,'log')
        % scale_fun = @(t) max(t+1,ceil(t.*log(t))); % scaling to use between m and n
        scale_fun = @(t) c_scale*max(t+1,ceil(t.*log(t)));
    elseif isequal(scale_type,'linear1')
        scale_fun = @(t) ceil(1.5*t);
    else
        scale_fun = @(t) ceil(2*t);
    end
    

    fprintf('\n========================================\n');
    fprintf('Running d = %d, OOD_type = %s\n', d, OOD_type);
    fprintf('========================================\n\n');

        for i = 1:length(a_val_vec)
            a_val = a_val_vec(i);
            for j = 1:length(b_val_vec)
                b_val = b_val_vec(j);
        
                if OOD_type == "hypercube"
                    w = ones(1,d).*b_val;
                elseif OOD_type == "unbounded"
                    w = (1:d).^b_val;
                else
                    error('Unknown OOD type: %s', OOD_type);
                end
        
                % err_grid = w.*(2*rand(K,d)-1);
                err_grid = 2*rand(K,d)-1;
                err_grid = [err_grid ; -ones(1,d)];
                err_grid = w.*err_grid;
        
                %% Main loop %%
        
                b_err_grid = func(err_grid, a_val)/sqrt(K);
        
                % Arrays for storing the raw experiment data.
                L2_err_data = zeros(num_trials,m_max+1);
                Linf_err_data = zeros(num_trials,m_max+1);
                cond_num_data = zeros(num_trials,m_max+1);
                m_vals_data = zeros(num_trials,m_max+1);
                n_vals_data = zeros(num_trials,m_max+1);
                kappa_data = zeros(num_trials,m_max+1);
        
                % Loop over the trials.
                parfor t = 1:num_trials
        
                    S = zeros(d,1);
                    n = 1;
                    u = 1;
        
                    L2_err_data_single = zeros(1,m_max+1);
                    Linf_err_data_single = zeros(1,m_max+1);
                    cond_num_data_single = zeros(1,m_max+1);
                    m_vals_data_single = zeros(1,m_max+1);
                    n_vals_data_single = zeros(1,m_max+1);
                    kappa_data_single = zeros(1,m_max+1);
        
                    % Find the maximum n given the scaling.
                    a = 1:m_max;
                    a = scale_fun(a);
                    n_max = find(a<=m_max,1,'last');
        
                    while n < n_max
        
                        n = size(S,2);
                        m = scale_fun(n);
        
                        n_vals_data_single(u) = n;
                        m_vals_data_single(u) = m;
        
                        A_err_grid = generate_measurement_matrix(poly_type,S,err_grid);
        
                        %%% Generate measurement matrix, sample points and vector %%%
                        y_grid = 2*rand(m,d)-1;
                        A = generate_measurement_matrix(poly_type,S,y_grid);
                        b = func(y_grid, a_val)/sqrt(m);
        
                        %%% Compute least-squares fit %%%
                        c = A\b;
        
                        %%% Evaluate on error grid %%%
                        bapprox_err_grid = A_err_grid*c;
                        L2_err = norm(bapprox_err_grid - b_err_grid)/norm(b_err_grid);
                        Linf_err = max(abs(bapprox_err_grid - b_err_grid))/max(abs(b_err_grid));
        
                        L2_err_data_single(u) = L2_err;
                        Linf_err_data_single(u) = Linf_err;
                        kappa_data_single(u) = cond(A_err_grid);
        
                        disp(['ALS: ',fun_name,space,' d = ',num2str(d),space, ...
                            ' trial = ',num2str(t),space,' n = ',num2str(n),space, ...
                            ' m = ',num2str(m),space,' Err (inf) = ',num2str(Linf_err)]);
        
                        %%% Update the set S %%%
                        RS = find_margin(S);
        
                        % Compute coefficients in reduced margin.
                        B = generate_measurement_matrix(poly_type,RS,y_grid);
                        fcoeffs = (b-A*c)'*B;
        
                        % Bulk procedure.
                        fsum = sum(abs(fcoeffs).^2);
                        [fcoeffs_sort,L] = sort(abs(fcoeffs).^2,'descend');
        
                        fpsum = 0;
                        ii = 0;
                        while fpsum < beta*fsum
                            ii = ii+1;
                            fpsum = fpsum + fcoeffs_sort(ii);
                        end
        
                        F = RS(:,L(1:ii));
        
                        S = [S F];
                        n = size(S,2);
        
                        u = u+1;
                    end
        
                    L2_err_data(t,:) = L2_err_data_single;
                    Linf_err_data(t,:) = Linf_err_data_single;
                    n_vals_data(t,:) = n_vals_data_single;
                    m_vals_data(t,:) = m_vals_data_single;
                    kappa_data(t,:) = kappa_data_single;
                end
        
        
                %% Best n-term approximation (disabled) %%
                best_m_vals = [];
                best_Linf_vals = [];
                %% Multiple-trial geometric mean and uncertainty region %%
                m_vals_all = [];
                mean_vals_all = [];
                curve_min_vals_all = [];
                curve_max_vals_all = [];
        
                for m = 1:m_max
                    I = find(m_vals_data == m);
                    data_m = Linf_err_data(I);
        
                    if ~isempty(data_m)
                        geo_mean = 10^mean(log10(data_m));
                        geo_std = std(log10(data_m));
                        curve_min = 10^(log10(geo_mean) - geo_std);
                        curve_max = 10^(log10(geo_mean) + geo_std);
        
                        m_vals_all = [m_vals_all m];
                        mean_vals_all = [mean_vals_all geo_mean];
                        curve_min_vals_all = [curve_min_vals_all curve_min];
                        curve_max_vals_all = [curve_max_vals_all curve_max];
                    end
                end
        
                %% Theoretical curves %%
                % Fitted convergence curves are intentionally NOT computed here.
                % They are computed from the saved data in the Python plotting script.
        
                if OOD_type == "unbounded"
                    if all(w == 1)
                        p_min = 1/2;
                    else
                        k = 1:d;
                        F = @(s) sum(arrayfun(@(k) ...
                            ((w(k) + sqrt(w(k)^2 - 1))^s - 1) ...
                            / (2*k^a_val), 1:d)) - 1;
                    
                        s_star = fzero(F, 0);
                        p_min = 2 / (1 + s_star);
                    end            
                    q_ana = 1/p_min - 1;
                    c_ana = max(mean_vals_all.*(m_vals_all.^(-q_ana)));
                    conv_est = c_ana*m_vals_all.^(q_ana);
                end
                xi_w = b_val+sqrt(b_val.^2-1);
                p = 2 / (1 + log(1 + 1 / sum((1:d).^(-2)/2)) / log(xi_w));
                q_ana2 = 1/p - 1;
                c_ana_2 = max(mean_vals_all.*(m_vals_all.^(-q_ana2)));
                conv_the = c_ana_2*m_vals_all.^(q_ana2);
        
                %% Save all data required for Python plotting %%
                data_dir = '../data';
                if ~exist(data_dir, 'dir')
                    mkdir(data_dir);
                end
        
                % best_m_vals = m_best(I_b);
                % best_Linf_vals = Linf_error_nbest(I_b);
        
                a_tag = strrep(strrep(num2str(a_val, '%.6g'), '.', 'p'), '-', 'm');
                b_tag = strrep(strrep(num2str(b_val, '%.6g'), '.', 'p'), '-', 'm');
        
                data_name = sprintf( ...
                    'ALS_%s_d%d_a%s_b%s_%s.mat', ...
                    fun_name, d, a_tag, b_tag, char(OOD_type));
        
                if OOD_type == "hypercube"
                    save(fullfile(data_dir, data_name), ...
                        ... % Experiment information
                        'fun_name', 'poly_type', 'd', 'a_val', 'b_val', ...
                        'w', 'OOD_type', 'm_max', 'num_trials', 'K', ...
                        ... % Raw trial data
                        'm_vals_data', 'n_vals_data', ...
                        'L2_err_data', 'Linf_err_data', 'kappa_data', ...
                        ... % Mean and uncertainty region
                        'm_vals_all', 'mean_vals_all', ...
                        'curve_min_vals_all', 'curve_max_vals_all', ...
                        ... % Best n-term curve
                        'best_m_vals', 'best_Linf_vals', ...
                        ... % ID theoretical curve
                        'q_ana2', 'c_ana_2', 'conv_the');
        
                elseif OOD_type == "unbounded"
                    save(fullfile(data_dir, data_name), ...
                        ... % Experiment information
                        'fun_name', 'poly_type', 'd', 'a_val', 'b_val', ...
                        'w', 'OOD_type', 'm_max', 'num_trials', 'K', ...
                        ... % Raw trial data
                        'm_vals_data', 'n_vals_data', ...
                        'L2_err_data', 'Linf_err_data', 'kappa_data', ...
                        ... % Mean and uncertainty region
                        'm_vals_all', 'mean_vals_all', ...
                        'curve_min_vals_all', 'curve_max_vals_all', ...
                        ... % Best n-term curve
                        'best_m_vals', 'best_Linf_vals', ...
                        ... % Unbounded theoretical curve
                        'q_ana', 'c_ana', 'conv_est', ...
                        ... % ID theoretical curve
                        'q_ana2', 'c_ana_2', 'conv_the');
                else
                    error('Unknown OOD type: %s', OOD_type);
                end
        
                fprintf('Experiment data saved: %s\n', ...
                    fullfile(data_dir, data_name));
            end
        end
    end     % b_val loop
    