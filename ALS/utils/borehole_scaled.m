function b = borehole_scaled(y)

range = [0.05 0.15 ; 100 50000 ; 63070 115600 ; 990 1110 ; 63.1 116 ; 700 820 ; 1120 1680 ; 9855 12045 ];
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
    
    b(i) = borehole(x);
    
end


end