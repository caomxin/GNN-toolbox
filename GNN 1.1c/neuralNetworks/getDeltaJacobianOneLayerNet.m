function [gradient,dInputs]=getDeltaJacobianOneLayerNet(net,netState,delta,jacNodes)
exampleNum=size(netState.inputs,2);

outFirstDer=1-netState.outs .*netState.outs;
outSecondDer=-2*netState.outs .* (1-netState.outs .*netState.outs);

on=delta .* outSecondDer .* net.weights1(:,jacNodes);

gradient.bias1=sum(on,2);   %(sum( on' ))'
gradient.weights1=on * netState.inputs';

df= outFirstDer .* delta;

for i=1:exampleNum
    gradient.weights1(:,jacNodes(i))=gradient.weights1(:,jacNodes(i))+df(:,i);
end
