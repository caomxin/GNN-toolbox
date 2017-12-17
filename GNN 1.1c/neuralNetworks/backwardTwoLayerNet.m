%function [gradient,dInputs]=backwardTwoLayerNet(net,netState,delta,saturationControl,networkType)
function [gradient,dInputs]=backwardTwoLayerNet(net,netState,delta)

global dynamicSystem

dnet2=delta .* (1-netState.outs.*netState.outs);

if dynamicSystem.config.useSaturationControl
    absval=abs(netState.outs)-dynamicSystem.config.saturationThreshold;
    absval(absval<0)=0;
    dnet2 = dnet2 + dynamicSystem.config.saturationCoeff.*absval.*sign(netState.outs);
end

gradient.weights2=dnet2*netState.hiddens';
gradient.bias2=sum(dnet2,2);
dnet1=(net.weights2'*dnet2) .* (1-netState.hiddens.*netState.hiddens);

if dynamicSystem.config.useSaturationControl
    absval=abs(netState.hiddens)-dynamicSystem.config.saturationThreshold;
    absval(absval<0)=0;
    dnet1 = dnet1 + dynamicSystem.config.saturationCoeff.*absval.*sign(netState.hiddens);
end

gradient.weights1=dnet1*netState.inputs';
gradient.bias1=sum(dnet1,2);
dInputs=net.weights1'*dnet1;
