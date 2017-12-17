%% This function just calls backwark a given number of times and accumulate
%% gradient in dPar
%function [dPar,dX,i]=neuralModelRunBackward(maxIt,x,dataset,p,delta,forwardState,sys,stopCoef)
function [dPar,dX,i]=neuralModelRunBackward(delta,forwardState,maxIt)

global dataSet dynamicSystem learning

xdim=dynamicSystem.config.nStates;

%jacobian=feval(sys.forwardJacobianFunction,dataSet.trainSet,dynamicSystem.parameters,learning.current.forwardState,sys);
[learning.current.jacobian, learning.current.jacobianErrors] = feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',forwardState);

dX=delta(:);
totDeltaX=zeros(size(dX));

if isempty(maxIt)
    maxIt=learning.config.maxBackwardSteps;
end
for i=1:maxIt
    totDeltaX=totDeltaX+dX;
    dX=learning.current.jacobian' * dX;
    stabCoefficient=sum(sum(abs(dX))) /sum(sum(abs(totDeltaX)));
    if(stabCoefficient < learning.config.backwardStopCoefficient) || (sum(sum(abs(totDeltaX))) == 0)
        break;
    end
end


dPar.transitionNet=feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet,...
    learning.current.forwardState.transitionNetState,reshape(totDeltaX'*dataSet.trainSet.neuralModel.childToArcMatrix',xdim,dataSet.trainSet.nArcs));
