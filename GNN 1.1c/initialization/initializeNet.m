function [parameters,net]=initializeNet(net)

switch net.nLayers
    case {1} 
        parameters.weights1=2*(rand(net.nOuts,net.nInputs)-0.5)*net.weightRange;
        parameters.bias1=2*(rand(net.nOuts,1)-0.5)*net.weightRange;
        if strcmp(net.outActivationType,'linear')
            net.forwardFunction=@forwardOneLayerLinearOutNet;
            net.backwardFunction=@backwardOneLayerLinearOutNet;
            net.getDeltaJacobianFunction=@getDeltaJacobianOneLayerLinearOutNet;
        elseif strcmp(net.outActivationType,'tanh')
            net.forwardFunction=@forwardOneLayerNet;
            net.backwardFunction=@backwardOneLayerNet;
            net.getDeltaJacobianFunction=@getDeltaJacobianOneLayerNet;
        end
    case {2}
        parameters.weights1=2*(rand(net.nHiddens,net.nInputs)-0.5)*net.weightRange;
        parameters.bias1=2*(rand(net.nHiddens,1)-0.5)*net.weightRange;
        parameters.weights2=2*(rand(net.nOuts,net.nHiddens)-0.5)*net.weightRange;
        parameters.bias2=2*(rand(net.nOuts,1)-0.5)*net.weightRange;
        if strcmp(net.outActivationType,'linear')
            net.forwardFunction=@forwardTwoLayerLinearOutNet;
            net.backwardFunction=@backwardTwoLayerLinearOutNet;
            net.getDeltaJacobianFunction=@getDeltaJacobianTwoLayerLinearOutNet;
        elseif strcmp(net.outActivationType,'tanh')
            net.forwardFunction=@forwardTwoLayerNet;
            net.backwardFunction=@backwardTwoLayerNet;
            net.getDeltaJacobianFunction=@getDeltaJacobianTwoLayerNet;
        end
end