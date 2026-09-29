%--- Description ---%
%
% Filename: aniso_exp.m
% Authors: Ben Adcock and Simone Brugiapaglia
% Part of the paper "Is Monte Carlo a bad sampling strategy for learning
% smooth functions in high dimensions?"
%
% Description: XXXXXX
%
% Input:
% y - m x d array of sample points
%
% Output:
% b - m x 1 array of function values at the sample points

function b = aniso_exp(y)

[m,d] = size(y);

b = zeros(m,1);
for k = 1:d
    b = b+y(:,k)/(2*k);
end
b = exp(b);

end