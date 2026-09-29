function run_best_n_term(poly_type,fun_name,r,d,sigma,additive,n_max_des,err_grid)

K = size(err_grid,1);
space = ' ';

func = str2func(fun_name); % convert function name to function handle

disp(['best n-term: ',fun_name,space,' d = ',num2str(d)]);
disp('Computing coefficients and index set:');
%%% Compute coefficients %%%
if additive == 0
    [f_coeffs,~,I] = compute_largest_coeffs(fun_name,r,d,sigma);
else
    [f_coeffs,~,I] = compute_largest_coeffs_additive(fun_name,d,sigma);
end

%%% Compute best n-term approximation error %%%

n_max = min(length(f_coeffs),n_max_des);
n_vals = 1:n_max;

% generate function values over the error grid
if additive == 0
    
    Gvals = zeros(K,r,d);
    for i = 1:r
        for j = 1:d
            Gvals(:,i,j) = func(i,j,r,d,err_grid(:,j));
        end
    end
    b_err_grid = zeros(K,1);
    for i = 1:r
        b_err_grid = b_err_grid + prod(Gvals(:,i,:),3)/sqrt(K);
    end
    
else
    
    b_err_grid = zeros(K,1);
    for i = 1:d
        b_err_grid = b_err_grid + func(i,d,err_grid(:,i))/sqrt(K);
    end
    
end

disp(['Computing error:']);

L2_error_data = [];
I = I(:,1:n_max);
A_full = generate_measurement_matrix(poly_type,I,err_grid);

res = -b_err_grid;

for n = n_vals
    
    a = A_full(:,n);
    res = res + f_coeffs(n)*a;
    
    % compute relative L2 error
    L2_err = norm(res)/norm(b_err_grid);
    L2_error_data = [L2_error_data L2_err];
    
    disp([fun_name,space,' d = ',num2str(d),space,' n = ',num2str(n),space,' Err = ',num2str(L2_err)]);
    
end

% save data
file_name = ['best_n_term_',fun_name,'_d',num2str(d)];
L2_error_data_nbest = L2_error_data;
save(['../data/', file_name, '.mat'], 'n_vals', 'L2_error_data_nbest');
% save(['../data/',file_name,'.mat'],'n_vals','L2_error_data');

end
