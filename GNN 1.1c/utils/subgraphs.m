function [n,r]=subgraphs(A,T,AS,TS)
tsdim=size(TS,2);
gstop=0;
n=zeros(size(A,1),1);
for j=1:tsdim
    pl=find(ismember(T',TS(:,j)','rows'));
    dims(j)=size(pl,1);
    if (dims(j)==0),
        gstop=1;
        break;
    end
    l(j,1:dims(j))=pl';
end

r=0;
ind=ones(tsdim,1);
v=l(:,1);
while ~gstop
    if (A(v,v))==AS
        r=r+1;
        n(v)=1;
    end

    i=tsdim;
    stop=0;
    while ~stop
        if (ind(i)+1)<=dims(i)
            ind(i)=ind(i)+1;
            v(i)=l(i,ind(i));
            stop=1;
        else
            if i>1
                ind(i)=1;
                v(i)=l(i,ind(i));
                i=i-1;
            else
                stop=1;
                gstop=1;
            end
        end
    end
end

