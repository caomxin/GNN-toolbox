%function [f,netState]=forwardTwoLayerNet(x,net)
function netState=forwardTwoLayerNet(x,net,optimalParam)

global dynamicSystem learning

%%%%%%% optimization: %%%%%%%%%%
%   kron(ones(m,n),A) -> repmat(A, [m n])

% o=tanh(net.weights1*x+kron(ones(1,sx),net.bias1));
% f=tanh(net.weights2*o+kron(ones(1,sx),net.bias2));
% netState.hiddens=o;

%% 1st version
%netState.hiddens=tanh(net.weights1*x+repmat(net.bias1, [1 sx]));
%f=tanh(net.weights2*netState.hiddens+repmat(net.bias2, [1 sx]));
%netState.outs=f;

%% 2nd version (new parameters)
sx=size(x,2);
if ~optimalParam
    netState.hiddens=tanh(dynamicSystem.parameters.(net).weights1*x+repmat(dynamicSystem.parameters.(net).bias1, [1 sx]));
    netState.outs=tanh(dynamicSystem.parameters.(net).weights2*netState.hiddens+repmat(dynamicSystem.parameters.(net).bias2, [1 sx]));
else
    netState.hiddens=tanh(learning.current.optimalParameters.(net).weights1*x+repmat(learning.current.optimalParameters.(net).bias1, [1 sx]));
    netState.outs=tanh(learning.current.optimalParameters.(net).weights2*netState.hiddens+repmat(learning.current.optimalParameters.(net).bias2, [1 sx]));
end
netState.inputs=x;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
