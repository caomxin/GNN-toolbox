function [jacobian]=forwardJacobianOneLayerNet(net,netState,ind)
dnet1=delta .* (1-netState.outs .* netState.outs);
gradient.weights1=dnet1*netState.hiddens';
gradient.bias1=sum(dnet1,2);    %sum(dnet1')'
dInputs=net.weights1'*delta;
