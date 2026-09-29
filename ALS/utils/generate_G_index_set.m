%--- Description ---%
%
% Inputs: 
% G - an d x N array of nonnegative numbers, where max(G(i,:)) <= 1 for
% each i
% sigma - tolerance
% verbose - either 0 (no output) or 1 (output)
% 
% Output:
% I - Multi-index set defined by G

function I = generate_G_index_set(G,sigma,verbose)

[d,~] = size(G);

I = find(G(1,:)>=sigma)-1;

if d >= 2
    
    for k = 2:d
        J = [];
        
        M = size(I,2);
        
        if verbose == 1
            disp(['k = ',num2str(k),' current index set size = ',num2str(M)]);
        end
        
        for i = 1:M
            z = I(:,i);
            
            c = 1;
            for l = 1:(k-1)
                c = c*G(l,z(l)+1);
            end
            
            S = find(G(k,:)>=sigma/c);
            K = repmat(z,1,length(S));
            J = [J [K ; S-1]];
            
        end
     
        I = J;
    end
    
    if verbose == 1
        disp(['k = ',num2str(k),' final index set size = ',num2str(size(I,2))]);
    end
    
end

end



