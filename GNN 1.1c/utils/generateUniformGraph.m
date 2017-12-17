function [g,failed]=  generateUniformGraph(d,s)

g=sparse([2:d,d,1:(d-1),1],[1:(d-1),1,2:d,d],ones(2*d,1),d,d);

exch=0;
if(s> d/2 && (mod(s,2)==1 || mod(d,2)==0)),
    s=d-s;
    exch=1;
end

failed=0;
for i=1:d
    toAdd=s-sum(g(:,i));
    nav=find(sum(g)==s);
    av=setdiff(setdiff(find(g(:,i)==0),(1:i)'),nav);
    if (toAdd>length(av))
        failed=1;
        break
    end
    if(toAdd>0),
        av=av(randperm(length(av)));    %new
        rp=[];                          %new
        for p=2:s-1                     %new
            pp=find(sum(g(:,av))==p);   %new
            rp(end+1:end+length(pp))=av(pp);    %new
            if length(rp) == length(av) %new
                break                   %new
            end                         %new
        end                             %new
%         rp=randperm(length(av));      %old
%         g(av(rp(1:toAdd)),i)=1;       %old
%         g(i,av(rp(1:toAdd)))=1;       %old
         g(rp(1:toAdd),i)=1;
         g(i,rp(1:toAdd))=1; 
    end
end

if(exch) 
    g=(1-g)-eye(d);
end