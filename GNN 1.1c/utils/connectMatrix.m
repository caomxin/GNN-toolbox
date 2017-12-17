function C=connectMatrix(M)

sm=size(M);
R=eye(size(M));
for i=1:sm(1)
    R=R+R*M;
end
[r,c]=find((R==0));
sc=size(c);
while sc(1)>0
    i=round(rand*(sc(1)-1)+1);
    M(r(i),c(i))=1;
    M(c(i),r(i))=1;
    sm=size(M);
    R=eye(size(M));
    for i=1:sm(1)
        R=R+R*M;
    end
    [r,c]=find((R==0));
    sc=size(c);
end
C=M;