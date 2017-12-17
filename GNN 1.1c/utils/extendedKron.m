function r=extendedKron(dataset,weightMatrix,n)

global dataSet

%[ma,na] = size(connMatrix);
%[ma,na] = size(dataSet.(dataset).connMatrix);
ma=dataSet.(dataset).nNodes;

%[mb,nb] = size(weightMatrix);
mb = size(weightMatrix,1);

% [ia,ja,sa] = find(connMatrix);
[ia,ja,sa] = find(dataSet.(dataset).connMatrix);

[ib,jb,sb] = find(weightMatrix(:,1:n));

%ia = ia(:); ja = ja(:); sa = sa(:);    %inutili
%ib = ib(:); jb = jb(:); sb = sb(:);

sizesb=size(sb);
sizesa_t=[size(sa,2) size(sa,1)];

%ka = ones(size(sa));
%kb = ones(size(sb));

t = mb*(ia-1)';

%%%%%%%% OPTIMIZATION %%%%%%%%%%%%
% A(ones(k,1),:) --> repmat(A(1,:),[k 1])

%ik = t(kb,:)+ib(:,ka);
ik = repmat(t(1,:),sizesb) + repmat(ib(:,1), sizesa_t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = n*(ja-1)';

%jk = t(kb,:)+jb(:,ka);
jk = repmat(t(1,:),sizesb) + repmat(jb(:,1), sizesa_t);


ik=ik(:);
jk=jk(:);

% colDiv=floor((jk-1) ./ n);    %mi sembrano inutili dato che (colMod+colDiv*n)=(jk-1) !!
% colMod=mod(jk-1,n);
rowMod=mod(ik-1,mb);
%res=weightMatrix(rowMod+1+(colMod +colDiv*n)*mb);

res=weightMatrix(rowMod+1+(jk-1)*mb);

%r = sparse(ik,jk,res(:),ma*mb,na*n);
r = sparse(ik,jk,res(:),ma*mb,ma*n);

