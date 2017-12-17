% function [e,outState]=rankingComputeError(x,trainSet,p,sys)
function [e,outState]=rankingComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem

%in=[x;trainSet.nodeLabels];

%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
else
    in=[x;dataSet.(dataset).nodeLabels];
end


%[outState.out,outState.outNetState]=feval(sys.outNet.forwardFunction,in,p.outNet);
outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);


% alpha value has been set experimentally
alpha=70;

%supervisedNodes=find(diag(trainSet.maskMatrix));
supervisedNodes=find(diag(dataSet.(dataset).maskMatrix));

%y=outState.out(supervisedNodes)';
y=outState.outNetState.outs(supervisedNodes)';

%e=errLogSig(alpha,trainSet.m,y);
e=errLogSig(alpha,dataSet.(dataset).m,y);

% outState.delta=zeros(1,size(trainSet.maskMatrix,1));
% outState.delta(supervisedNodes)=gradErrLogSig(alpha,trainSet.m,y);
outState.delta=zeros(1,size(dataSet.(dataset).maskMatrix,1));
outState.delta(supervisedNodes)=gradErrLogSig(alpha,dataSet.(dataset).m,y);
