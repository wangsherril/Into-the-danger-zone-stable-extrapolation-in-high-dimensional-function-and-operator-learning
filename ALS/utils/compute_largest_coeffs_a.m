% Computes the mult-index set I and coefficients larger than sigma for a
% separable function
%
% CURRENTLY ONLY IMPLEMENTED FOR LEGENDRE COEFFICIENTS!!

function [f_coeffs,S,I] = compute_largest_coeffs_a(fun_name,r,d,sigma, a_val)

func = str2func(fun_name);

%%% Compute the coefficients of the r*d 1D functions

N = 0;
for i = 1:r
    for j = 1:d
        gij = chebfun(@(x) func(i,j,r,d,x, a_val));
        N = max(N,length(gij));
    end
end

G_coeffs = zeros(r,d,N);
for i = 1:r
    for j = 1:d
        gij = chebfun(@(x) func(i,j,r,d,x, a_val));
        gij_cheb = chebcoeffs(gij);
        gij_leg = cheb2leg(gij_cheb);
        n = length(gij_leg);
        scl = sqrt(2*(0:(n-1))+1);
        gij_leg = gij_leg./scl';
        G_coeffs(i,j,1:n) = gij_leg';
    end
end

% [x,w] = lgwt(quad_order+1,-1,1); % compute Legendre Gaussian quadrature nodes and weights
% G_coeffs = zeros(r,d,quad_order+1); % array for coefficients of functions g1,...,gd
% B = legmat(x,quad_order+1);
% for i = 1:r
%     for j = 1:d
%         gjval = func(i,j,r,d,x);
%         for l = 1:(quad_order+1)
%             gamma = sum((B(:,l).^2).*w);
%             G_coeffs(i,j,l) = sum(gjval.*B(:,l).*w)/gamma;
%         end
%     end
% end

%%% Compute upper bound G
G = zeros(d,N);
for j = 1:d
    for l = 1:N
        G(j,l) = (r^(1/d))*max(abs(G_coeffs(:,j,l)));
    end
end

%%% Renormalize G and adjust sigma
sigma2 = sigma;
for j = 1:d
    g = max(G(j,:));
    G(j,:) = G(j,:)/g;
    sigma2 = sigma2/g;
end

%%% Compute large index set I
I = generate_G_index_set(G,sigma2,1);

%%% Compute coefficients in I
num_coeffs = size(I,2);
f_coeffs = zeros(num_coeffs,1);

for l = 1:num_coeffs
    for i = 1:r
        b = zeros(d,1);
        for j = 1:d
            b(j) = G_coeffs(i,j,I(j,l)+1);
        end
        f_coeffs(l) = f_coeffs(l) + prod(b);
    end
end

[c,S] = sort(abs(f_coeffs),'descend');
n_max = find(c >= sigma,1,'last');
S = S(1:n_max);
f_coeffs = f_coeffs(S);
I = I(:,S);

end


