function [f,netState]=forwardJacobianOneLayerLinearOutNet(x,net)
sx=size(x);
f=net.weights1*x+kron(ones(1,sx(2)),net.bias1);
netState.outs=f;
netState.inputs=x;
