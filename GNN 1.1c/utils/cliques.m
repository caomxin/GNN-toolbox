function [ind, num] = cliques (A, n)
% Find cliques of dimension n in graph with connection matrix A
% A: connection matrix
% n: clique dimension
% ind: indexes of nodes that forms cliques
% num: number of cliques found
C=ones(n,n)&~eye(n);
dim = size(A,1);
num=0;
ind=zeros(dim,1);
indexes=[1:n];
stop=0;
carry=0;
while ~stop
    if A(indexes,indexes)==C
        num=num+1;
        ind(indexes)=1;
    end
    % next try
    indexes(n)=indexes(n)+1;
    if indexes(n)>dim
        carry=1;
        cur=n-1;
        while carry
            if cur==0;
                stop=0;
                break;
            end
            indexes(cur)=indexes(cur)+1;
            if indexes(cur) <= dim+cur-n
                carry=0;
            else
                cur=cur-1;
            end
        end
        if cur==0
            break
        else
            for i=cur+1:n
                val=indexes(cur);
                indexes(i)=val+i-cur;
            end
        end
    end
end


