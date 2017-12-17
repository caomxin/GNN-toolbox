function areIsomorph(A1,A2)

% check iniziali: stesso numero nodi, numero compatibile di nodi con lo
% stesso numero di figli. Supponiamo di sì per ora

%A,B versioni stocastiche
h=sum(A1);
for i=1:size(A1,2)
    A(:,i)=A1(:,i)/h(i);
end
h=sum(A2);
for i=1:size(A2,2)
    B(:,i)=A2(:,i)/h(i);
end

%calcola pageRank con d=0.85 and tol=1E-8
d=0.85;
tol=1E-8;

n=size(A,2);
forcing=((1-d)/n)*ones(n,1);
prA=forcing;
prAold=zeros(n,1);

%i=0;
while(max(abs(prA-prAold))>tol)
    prAold=prA;
    prA = d*A*prA + forcing;
    %i=i+1;
end

prB=forcing;
prBold=zeros(n,1);

%i=0;
while(max(abs(prB-prBold))>tol)
    prBold=prB;
    prB = d*B*prB + forcing;
    %i=i+1;
end

uniqA=unique(prA);
uniqB=unique(prB);
szuA=size(uniqA,1);
szuB=size(uniqB,1);

if (szuA==n && szuB==n)
    if (max(abs(uniqA-uniqB)) < tol)
        disp('Sono isomorfi. Vettori di pageRank con valori distinti e uguali')
    else
        disp('Non sono isomorfi. Vettori di pageRank con valori distinti ma diversi')
    end
else
    if szuA~=szuB
        disp('Non sono isomorfi. Vettori di pageRank con valori non distinti e diversi')
    else
        % dividiamo in classi PR-equivalenti
        clA=zeros(szuA,n-1);
        for i=1:szuA
            clA(i,1:size(find(prA==uniqA(i)),1))=find(prA==uniqA(i));
        end
        clB=zeros(szuB,n-1);
        for i=1:szuB
            clB(i,1:size(find(prB==uniqB(i)),1))=find(prB==uniqB(i));
        end



        disp('Boh?')
        szuA
        szuB
        if szuA==szuB
            max(abs(uniqA-uniqB))
        else
            prA'
            uniqA'
            prB'
            uniqB'
        end
        clA
        clB
    end
end





