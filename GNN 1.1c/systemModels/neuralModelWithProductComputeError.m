function [e,outState]=neuralModelWithProductComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem learning

% in=[x;dataSet.(dataset).nodeLabels];
%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
    firstComponent=dynamicSystem.state(1,:);
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
    firstComponent=learning.current.validationState(1,:);    
else
    in=[x;dataSet.(dataset).nodeLabels];
    firstComponent=x(1,:);
end

outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);
%outState.out=out .* x(1,:); % with product!!
outState.out=outState.outNetState.outs .* firstComponent; % with product!!

%% Compute the error. The error is the quadratic difference of the targets from current outputs
%% In general  supervision may be placed only on some outputs: matrix
%% "maskMatrix" allows to select the supervised outputs
% outState.delta=(dataSet.(dataset).maskMatrix*(outState.out-dataSet.(dataset).targets)')';
outState.delta=(dataSet.(dataset).maskMatrix*(outState.out-dataSet.(dataset).targets)')';

e=outState.delta(:)' *outState.delta(:)/2;
