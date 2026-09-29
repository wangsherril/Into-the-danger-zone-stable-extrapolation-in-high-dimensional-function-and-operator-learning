function b = separable_nonintegrable_singularity_delta_ia(i,j,r,d,y,a_val)

delta = j.^(a_val);
b = sqrt(2*delta+delta^2)./(y+1+delta);

end