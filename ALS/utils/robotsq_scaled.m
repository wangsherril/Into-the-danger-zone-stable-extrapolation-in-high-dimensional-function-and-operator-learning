function b = robotsq_scaled(y)

range = [0 2*pi ; 0 2*pi ; 0 2*pi ; 0 2*pi ; 0 1 ; 0 1; 0 1; 0 1];
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
    
    b(i) = robotsq(x);
    
end


end