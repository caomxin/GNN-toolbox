function [gradient,dInputs]=getDeltaJacobianOneLayerLinearOutNet(net,netState,delta,jacNodes)
sx=size(netState.inputs,1);
exampleNum=size(netState.inputs,2);
outNum=size(delta,1);

gradient.weights1=zeros(outNum,sx);
for i=1:exampleNum
    gradient.weights1(:,jacNodes(i))=gradient.weights1(:,jacNodes(i))+delta(:,i);
end
gradient.bias1=zeros(outNum,1);


