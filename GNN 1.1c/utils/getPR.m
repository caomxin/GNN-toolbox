function X=getPR(X,W,E,n)
for i=1:n
    X= W * X+E;
end