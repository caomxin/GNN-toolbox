function [gradient,dInputs]=backwardOneLayerLinearOutNet(net,netState,delta,saturationControl,networkType)
%networkType=0 --> comparisonNet,   networkType=1 --> GNN

%saturationControl e networkType sono inutilizzati perché il controllo di saturazione non ha senso qui. 
%Sono stati mantenuti per uniformità con le altre

gradient.weights1=delta*netState.inputs';
gradient.bias1=sum(delta,2);
dInputs=net.weights1'*delta;
