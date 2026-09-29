%--- Description ---%
%
% Filename: oned_fun.m
% Authors: Ben Adcock and Simone Brugiapaglia
% Part of the paper "Is Monte Carlo a bad sampling strategy for learning
% smooth functions in high dimensions?"
%
% Description: evaluates the function f considered in Fig. 6
%
% Input:
% y - m x d array of sample points
%
% Output:
% b - m x 1 array of function values at the sample points

function b = separable_oned_fun(i,d,y)

m = length(y);

if i == 1
b = 1./(10-9*y);
else
    b = zeros(m,1);
end

end
