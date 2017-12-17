%% This check gradient computation for layered network .
%% It is the samae as checkGradient, but it can be applied to check the neural
%% network gradient computation
checkNet.nInputs=5;
checkNet.nOuts=1;

checkNetSet.nSamples=20;
checkNetSet.inputs=2*rand(checkNet.nInputs,checkNetSet.nSamples)-1;
checkNetSet.targets=2*rand(checkNet.nOuts,checkNetSet.nSamples)-1;


checkNet.nHiddens=10;
checkNet.weightRange=0.01;

checkNet.parameters.weights1=5*(rand(checkNet.nHiddens,checkNet.nInputs)-0.5)*checkNet.weightRange;
checkNet.parameters.bias1=5*(rand(checkNet.nHiddens,1)-0.5)*checkNet.weightRange;
checkNet.parameters.weights2=5*(rand(checkNet.nOuts,checkNet.nHiddens)-0.5)*checkNet.weightRange;
checkNet.parameters.bias2=5*(rand(checkNet.nOuts,1)-0.5)*checkNet.weightRange;

checkNet.forwardFunction=@forwardTwoLayerNet;
checkNet.backwardFunction=@backwardTwoLayerNet;

[checkNetOut,checkNetState]=feval(checkNet.forwardFunction,checkNetSet.inputs,checkNet.parameters);
delta=checkNetOut-checkNetSet.targets;
error=sum(delta *delta')/2;

[gradient,dx]=feval(checkNet.backwardFunction,checkNet.parameters,checkNetState,delta);

analyticGradient=gradient;
analyticDX=dx;

smallNumber=10e-8;

%%
%% computing the gradient by small perturbations

clear experimentalGradient;
for it1=fieldnames(checkNet.parameters)'
    sp=size(checkNet.parameters.(char(it1)));
    it1
    for r=1:sp(1)
        for c=1:sp(2)
            P=checkNet.parameters;
            P.(char(it1))(r,c)=P.(char(it1))(r,c)+smallNumber;
            eo=feval(checkNet.forwardFunction,checkNetSet.inputs,P);
            eDelta=eo-checkNetSet.targets;
            experimentalError=sum(eDelta *eDelta')/2;
            experimentalGradient.(char(it1))(r,c)=(experimentalError-error)/smallNumber;
        end
    end
end

%%
%% computing deltaX by small perturbation

sx=size(checkNetSet.inputs);
experimentalDX=zeros(sx(1),sx(2));
for r=1:sx(1)
    for c=1:sx(2)
        cX=checkNetSet.inputs;
        cX(r,c)=cX(r,c)+smallNumber;
        eo=feval(checkNet.forwardFunction,cX,checkNet.parameters);
        eDelta=eo-checkNetSet.targets;
        experimentalError=sum(eDelta *eDelta')/2;
        experimentalDX(r,c)=(experimentalError-error)/smallNumber;
    end
end



%% comparing gradients


maxDiff=0;
sumDiff=0;
maxRelativeErrorWRTExperimental=0;
maxRelativeErrorWRTAnalytic=0;
maxComponentRelativeErrorWRTExperimental=0;
maxComponentRelativeErrorWRTAnalytic=0;
experimentalGradientNorm=0;
analyticGradientNorm=0;

for it1=fieldnames(experimentalGradient)'
    it1
    gradientDifference.(char(it1))=experimentalGradient.(char(it1))-analyticGradient.(char(it1));
    maxDiff=max(maxDiff,max(max(abs(gradientDifference.(char(it1))))));
    sumDiff=sum(sum(abs(gradientDifference.(char(it1)))))+sumDiff;
    relativeErrorWRTExperimental=max(max(abs(gradientDifference.(char(it1))) ./abs(experimentalGradient.(char(it1)))));
    maxRelativeErrorWRTExperimental=max(maxRelativeErrorWRTExperimental,relativeErrorWRTExperimental);
    relativeErrorWRTAnalytic=max(max(abs(gradientDifference.(char(it1))) ./abs(analyticGradient.(char(it1)))));
    maxRelativeErrorWRTAnalytic=max(maxRelativeErrorWRTAnalytic,relativeErrorWRTAnalytic);

    componentRelativeErrorWRTExperimental=sum(sum(abs(gradientDifference.(char(it1))))) /sum(sum(abs(experimentalGradient.(char(it1)))))
    maxComponentRelativeErrorWRTExperimental=max(maxComponentRelativeErrorWRTExperimental,componentRelativeErrorWRTExperimental);
    componentRelativeErrorWRTAnalytic=sum(sum(abs(gradientDifference.(char(it1))))) /sum(sum(abs(analyticGradient.(char(it1)))))
    maxComponentRelativeErrorWRTAnalytic=max(maxComponentRelativeErrorWRTAnalytic,componentRelativeErrorWRTAnalytic);

    experimentalGradientNorm=sum(sum(abs(experimentalGradient.(char(it1)))))+experimentalGradientNorm;
    analyticGradientNorm=sum(sum(abs(analyticGradient.(char(it1)))))+analyticGradientNorm;
end

maxDiff
sumDiff
maxRelativeErrorWRTExperimental
maxRelativeErrorWRTAnalytic

maxComponentRelativeErrorWRTExperimental
maxComponentRelativeErrorWRTAnalytic
globalRelativeErrorWRTExperimental= sumDiff / experimentalGradientNorm
globalRelativeErrorWRTAnalytic=sumDiff /analyticGradientNorm