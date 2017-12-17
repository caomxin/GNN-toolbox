function [e,outState]=neuralModelQuadraticComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem learning

%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
else
    in=[x;dataSet.(dataset).nodeLabels];
end


%[outState.out,outState.outNetState]=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet');
outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);

%% Compute the error. The error is the quadratic difference of the targets from current outputs
%% In general  supervision may be placed only on some outputs: matrix "maksMatrix" allows to select the supervised outputs
outState.delta=(dataSet.(dataset).maskMatrix*outState.outNetState.outs')';
e= outState.outNetState.outs*dataSet.(dataset).maskMatrix*outState.outNetState.outs'/2;
