function b = piston_scaled(y)

range = [30 60 ; 0.005 0.020 ; 0.002 0.010 ; 1000 5000 ; 90000 110000 ; 290 296 ; 340 360];
dmax = size(range,1);

[m,d] = size(y);

if d > dmax
    disp('WARNING: Input dimension exceeds number of parameters. Ignoring all extra dimensions.');
end

sc1 = (range(:,2)-range(:,1))/2;
sc2 = (range(:,2)+range(:,1))/2;

b = zeros(m,1);
for i = 1:m
    
    if d >= dmax
        z = y(i,1:dmax);
    else
        z = [y(i,:) ones(1,dmax-d)];
    end
    
    z = z';
    x = sc1.*z + sc2;
    
    b(i) = piston(x);
    
end


end