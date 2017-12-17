function [gradient,dInputs]=backwardOneLayerNet(net,netState,delta,saturationControl,networkType)
%networkType=0 --> comparisonNet,   networkType=1 --> GNN

global dynamicSystem comparisonNet

dnet1=delta .* (1-netState.outs .* netState.outs);

if saturationControl 
    if networkType %GNN
        absval=abs(netState.outs)-dynamicSystem.config.saturationThreshold;
        absval(absval<0)=0;
        dnet2 = dnet2 + dynamicSystem.config.saturationCoeff.*absval.*sign(netState.outs);
    else %comparisonNet
        absval=abs(netState.outs)-comparisonNet.saturationThreshold;
        absval(absval<0)=0;
        dnet2 = dnet2 + comparisonNet.saturationCoeff.*absval.*sign(netState.outs);
    end
end

gradient.weights1=dnet1*netState.inputs';
gradient.bias1=sum(dnet1,2);
dInputs=net.weights1'*dnet1;
