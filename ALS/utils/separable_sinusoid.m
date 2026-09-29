%--- Description ---%
%
% Filename: iso_exp.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse Polynomial Approximation of High-Dimensional
% Functions", SIAM, 2021
%
% Edited by: Ben Adcock and Simone Brugiapaglia
% Part of the paper "XXXXXX"
%
% Description: EDIT!!
%
% Inputs:
% i - function to evaluate
% d - dimension
% y - m x 1 array of 1D sample points
%
% Output:
% b - m x 1 array of function values of g_i at the sample points

function b = separable_sinusoid(i,d,y)

    b = 0.3 + sin(16*y/15 - 0.7) + (sin(16*y/15 - 0.7)).^2 ;

end