% Computes the mult-index set I and coefficients larger than sigma for a
% additively separable function
%
% CURRENTLY ONLY IMPLEMENTED FOR LEGENDRE COEFFICIENTS!!

function [f_coeffs,S,I] = compute_largest_coeffs_additive(fun_name,d,sigma)

func = str2func(fun_name);

%%% Compute nonzero coefficients of f

f_coeffs = [];
N_vals = [1];
f_mean = 0;
for i = 1:d
    gi = chebfun(@(x) func(i,d,x)); 
    gi_leg = cheb2leg(chebcoeffs(gi));
    n = length(gi_leg);
    scl = sqrt(2*(0:(n-1))+1);
    gi_leg = gi_leg./scl';
    
    f_mean = f_mean + gi_leg(1); 
    f_coeffs = [f_coeffs ; gi_leg(2:end)];
    N_vals = [N_vals ; length(gi_leg)-1];
end

%%% Find the coefficients exceeding sigma

f_coeffs = [f_mean ; f_coeffs];
[c,S] = sort(abs(f_coeffs),'descend');
n_max = find(c >= sigma,1,'last');
S = S(1:n_max);
f_coeffs = f_coeffs(S);

%%% Construct the corresponding multi-index set I

B = tril(ones(d+1));
N_cum = B*N_vals;

I = zeros(d,n_max);
for n = 1:n_max
    
    if S(n) ~= 1
    i = find(N_cum > S(n),1)-1;
    I(i,n) = S(n)-N_cum(i);
    end
    
end

end




