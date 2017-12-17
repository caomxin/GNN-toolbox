function [f,netState]=forwardOneLayerNet(x,net)
sx=size(x,2);

%%%%%%% optimization: %%%%%%%%%%
%   kron(ones(m,n),A) -> repmat(A, [m n])

%f=tanh(net.weights1*x+kron(ones(1,sx),net.bias1));
%netState.outs=f;

netState.outs=tanh(net.weights1*x+repmat(net.bias1, [1 sx]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

netState.inputs=x;
