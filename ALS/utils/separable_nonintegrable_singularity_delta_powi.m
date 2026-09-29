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
% Description: computes the ith function g_i defining the isotropic exponential
% function as f = g_1 * g_2 * ... * g_d
%
% Inputs:
% i - function to evaluate
% d - dimension
% y - m x 1 array of 1D sample points
%
% Output:
% b - m x 1 array of function values of g_i at the sample points

function b = separable_nonintegrable_singularity_delta_powi(i,j,r,d,y)

delta = 2^j;
b = sqrt(2*delta+delta^2)./(y+1+delta);

end