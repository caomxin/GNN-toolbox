function x=mygetPR(A,d)
n=size(A,2);
x=((1-d)/n)*ones(n,1);
xold=zeros(n,1);
h=sum(A);
for i=1:size(A,2)
    W(:,i)=A(:,i)/h(i);
end
%W=A/6;
i=0;
while(max(abs(x-xold))>1E-10)
    xold=x;
    x= d*W * x+((1-d)/n)*ones(n,1);
    i=i+1;
end
disp(['Number of iteration: ' num2str(i)]);