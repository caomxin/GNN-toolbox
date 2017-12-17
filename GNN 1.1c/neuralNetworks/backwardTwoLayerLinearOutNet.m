%function [gradient,dInputs]=backwardTwoLayerLinearOutNet(net,netState,delta,saturationControl,networkType)
function [gradient,dInputs]=backwardTwoLayerLinearOutNet(net,netState,delta)

global dynamicSystem comparisonNet

gradient.weights2=delta*netState.hiddens';

gradient.bias2=sum(delta,2);
dnet1=(net.weights2'*delta) .* (1-netState.hiddens.*netState.hiddens);

if dynamicSystem.config.useSaturationControl
    absval=abs(netState.hiddens)-dynamicSystem.config.saturationThreshold;
    absval(absval<0)=0;
    dnet1 = dnet1 + dynamicSystem.config.saturationCoeff.*absval.*sign(netState.hiddens);
end
gradient.weights1=dnet1*netState.inputs';

gradient.bias1=sum(dnet1,2);
dInputs=net.weights1'*dnet1;
