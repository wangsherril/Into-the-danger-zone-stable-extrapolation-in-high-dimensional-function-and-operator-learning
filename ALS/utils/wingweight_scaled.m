function b = wingweight_scaled(y)

range = [150 200 ; 220 300 ; 6 10 ; -10 10 ; 16 45 ; 0.5 1 ; 0.08 0.18 ; 2.5 6 ; 1700 2500 ; 0.025 0.08];
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
    
    b(i) = wingweight(x);
    
end


end