function f = nonintegrable_singularity_delta_i(y)

[~,d] = size(y);

delta = 1:d;

product = 1;
for k = 1:d
    product = product .*sqrt(2*delta(k)+delta(k)^2)./ (y(:,k) + 1 + delta(k));
end
f = product;

end

