%function [x,state,i]=neuralModelRunForward(maxIt,x,dataset,p,sys,stopCoef)
function [x,state,i]=neuralModelRunForward(maxSteps,x,dataset,optimalParam)
%% This function just call forward a number "n" of times and compute the new state.

global dataSet dynamicSystem learning

if (isfield(dynamicSystem.config,'useLabelledEdges') && (dynamicSystem.config.useLabelledEdges==1))
    labels=[dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.childOfArc);dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.fatherOfArc);dataSet.(dataset).edgeLabels];
else
    labels=[dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.childOfArc);dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.fatherOfArc)];
end

%for i=1:maxIt
for i=1:maxSteps
    in=[x(:,dataSet.(dataset).neuralModel.fatherOfArc);labels];

    %[y,state.transitionNet]=feval(sys.transitionNet.forwardFunction,in,p.transitionNet);
    %s=dataset.neuralModel.childToArcMatrix' *y(:);
    %state.transitionNet=feval(sys.transitionNet.forwardFunction,in,'transitionNet');
    state.transitionNetState=feval(dynamicSystem.config.transitionNet.forwardFunction,in,'transitionNet',optimalParam);

    %s=dataset.neuralModel.childToArcMatrix' *state.transitionNet.outs(:);
    s=dataSet.(dataset).neuralModel.childToArcMatrix' *state.transitionNetState.outs(:);
    nx=reshape(s,size(x));
    stabCoef=(sum(sum(abs(x-nx)))) / sum(sum(abs(nx)));

    x=nx;
    if(stabCoef<learning.config.forwardStopCoefficient)
        break;
    end
end
