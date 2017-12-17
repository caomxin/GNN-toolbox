function test4uniform

global dynamicSystem dataSet learning testing TestFigH

dataSet.testSet.forwardSteps=30;

%evaluating current parameters on trainset
x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,...
    'trainSet',0);
%% Compute the error
% [currentErrorOnTrain,currentOutState]=feval(dynamicSystem.config.computeErrorFunction,x,...
%     dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
[currentErrorOnTrain,currentOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);
testing.current.trainSet.results=sparse([],[],[],dataSet.config.maxGraph+1,1);
for i=1:size(dataSet.trainSet.graphRanges,1),
    no=sum(currentOutState.outNetState.outs(dataSet.trainSet.graphRanges(i,1): dataSet.trainSet.graphRanges(i,2))>0);
    gd=dataSet.trainSet.graphRanges(i,2)-dataSet.trainSet.graphRanges(i,1)+1;
    testing.current.trainSet.results((dataSet.config.maxGraph-gd)/2+1+no)=...
        testing.current.trainSet.results((dataSet.config.maxGraph-gd)/2+1+no)+1;
end
testing.current.trainSet.results=testing.current.trainSet.results/dataSet.trainSet.graphNum;






%evaluating current parameters on testset
x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',0);
%% Compute the error
% [currentErrorOnTest,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,x,...
%     dataSet.testSet,dynamicSystem.parameters,dynamicSystem.config);
[currentErrorOnTest,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,0);
testing.current.testSet.results=sparse([],[],[],dataSet.config.maxGraph+1,1);
for i=1:size(dataSet.testSet.graphRanges,1),
    no=sum(currentTestOutState.outNetState.outs(dataSet.testSet.graphRanges(i,1): dataSet.testSet.graphRanges(i,2))>0);
    gd=dataSet.testSet.graphRanges(i,2)-dataSet.testSet.graphRanges(i,1)+1;
    testing.current.testSet.results((dataSet.config.maxGraph-gd)/2+1+no)=...
        testing.current.testSet.results((dataSet.config.maxGraph-gd)/2+1+no)+1;
end
testing.current.testSet.results=testing.current.testSet.results/dataSet.testSet.graphNum;



%evaluating optimal parameters on trainset
x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.trainSet.nNodes),...
    'trainSet',1);
%% Compute the error
% [errorOnTrain,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,...
%     dataSet.trainSet,learning.current.optimalParameters,dynamicSystem.config);
[errorOnTrain,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,1);
testing.optimal.trainSet.results=sparse([],[],[],dataSet.config.maxGraph+1,1);
for i=1:size(dataSet.trainSet.graphRanges,1),
    no=sum(trainOutState.outNetState.outs(dataSet.trainSet.graphRanges(i,1): dataSet.trainSet.graphRanges(i,2))>0);
    gd=dataSet.trainSet.graphRanges(i,2)-dataSet.trainSet.graphRanges(i,1)+1;
    testing.optimal.trainSet.results((dataSet.config.maxGraph-gd)/2+1+no)=...
        testing.optimal.trainSet.results((dataSet.config.maxGraph-gd)/2+1+no)+1;
end
testing.optimal.trainSet.results=testing.optimal.trainSet.results/dataSet.trainSet.graphNum;



%evaluating optimal parameters on testset
x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',1);
%% Compute the error
% [errorOnTest,testOutState]=feval(dynamicSystem.config.computeErrorFunction,x,...
%     dataSet.testSet,learning.current.optimalParameters,dynamicSystem.config);
[errorOnTest,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);
testing.optimal.testSet.results=sparse([],[],[],dataSet.config.maxGraph+1,1);
for i=1:size(dataSet.testSet.graphRanges,1),
    no=sum(testOutState.outNetState.outs(dataSet.testSet.graphRanges(i,1): dataSet.testSet.graphRanges(i,2))>0);
    gd=dataSet.testSet.graphRanges(i,2)-dataSet.testSet.graphRanges(i,1)+1;
    testing.optimal.testSet.results((dataSet.config.maxGraph-gd)/2+1+no)=testing.optimal.testSet.results((dataSet.config.maxGraph-gd)/2+1+no)+1;
end
testing.optimal.testSet.results=testing.optimal.testSet.results/dataSet.testSet.graphNum;


% displays results
global VisualMode DisplayFigH
if VisualMode == 1
    sz=floor(size(testing.optimal.testSet.results,1)/2);
    sz1=floor(size(testing.optimal.trainSet.results,1)/2);
    TestFigH=DisplayTestU(testing.current.trainSet.results(sz1+1),...
        testing.current.trainSet.results(sz1)+testing.current.trainSet.results(sz1+1)+testing.current.trainSet.results(sz1+2),...
        1-(testing.current.trainSet.results(sz1)+testing.current.trainSet.results(sz1+1)+testing.current.trainSet.results(sz1+2)),...
        testing.optimal.trainSet.results(sz1+1),...
        testing.optimal.trainSet.results(sz1)+testing.optimal.trainSet.results(sz1+1)+testing.optimal.trainSet.results(sz1+2),...
        1-(testing.optimal.trainSet.results(sz1)+testing.optimal.trainSet.results(sz1+1)+testing.optimal.trainSet.results(sz1+2)),...
        testing.current.testSet.results(sz+1),...
        testing.current.testSet.results(sz)+testing.current.testSet.results(sz+1)+testing.current.testSet.results(sz+2),...
        1-(testing.current.testSet.results(sz)+testing.current.testSet.results(sz+1)+testing.current.testSet.results(sz+2)),...
        testing.optimal.testSet.results(sz+1),...
        testing.optimal.testSet.results(sz)+testing.optimal.testSet.results(sz+1)+testing.optimal.testSet.results(sz+2),...
        1-(testing.optimal.testSet.results(sz)+testing.optimal.testSet.results(sz+1)+testing.optimal.testSet.results(sz+2)));
end