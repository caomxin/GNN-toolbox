function test4classification(saveMem)
% testing function for classification problems (private)

global dataSet dynamicSystem learning testing    
if nargin==0
    saveMem=0;
end

dataSet.testSet.forwardSteps=30;

trainIndex=find(diag(dataSet.trainSet.maskMatrix));
validationIndex=find(diag(dataSet.validationSet.maskMatrix));
testIndex=find(diag(dataSet.testSet.maskMatrix));
supervisedNodesNumberTrain=size(trainIndex,1);
supervisedNodesNumberTest=size(testIndex,1);
supervisedNodesNumberValidation=size(validationIndex,1);


try
% evaluating current parameters on trainSet

if saveMem      % outOfMemory: keep only trainSet
    try
        save('data.mat','dataSet')
    catch
        err(0,'Can''t save data.mat in test4classification. Aborting...')
        return
    end
    dataSet.validationSet=[];
    dataSet.testSet=[];
    dataSet.testSet.forwardSteps=30;
    pack;
end

[x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',0);
[testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);

if size(dataSet.trainSet.targets,1)==1
    testing.current.trainSet.mistakenPatternIndex=find((dataSet.trainSet.targets>0 & ...
    currentTrainOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
    (dataSet.trainSet.targets<0 & currentTrainOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
    testing.current.trainSet.mistakenPatternIndex=intersect(testing.current.trainSet.mistakenPatternIndex,trainIndex);
else
    % the following instructions should take care of the pure multiclass problems. 
    % A pattern is correct if its maximum value is in the correct position
    [vv,ii]=max(dataSet.trainSet.targets(:,trainIndex),[],1);
    [v2,i2]=max(currentTrainOutState.outNetState.outs(:,trainIndex),[],1);
    testing.current.trainSet.mistakenPatternIndex=find(ii-i2);
end
evaluateAccuracyOnGraphs('trainSet','current')

%% save outputs, useful to evaluate equal error rate
testing.current.trainSet.out=currentTrainOutState.outNetState.outs;

testing.current.trainSet.mistakenPatterns=dataSet.trainSet.nodeLabels(:,testing.current.trainSet.mistakenPatternIndex);
testing.current.trainSet.mistakenTargets=dataSet.trainSet.targets(testing.current.trainSet.mistakenPatternIndex);
testing.current.trainSet.accuracy=1-(size(testing.current.trainSet.mistakenPatternIndex(:),1)/supervisedNodesNumberTrain);


% evaluating optimal parameters on trainSet
[x,trainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',1);
[testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,1);

if size(dataSet.trainSet.targets,1)==1
    testing.optimal.trainSet.mistakenPatternIndex=find((dataSet.trainSet.targets>0 & ...
    trainOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
    (dataSet.trainSet.targets<0 & trainOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
    testing.optimal.trainSet.mistakenPatternIndex=intersect(testing.optimal.trainSet.mistakenPatternIndex,trainIndex);
else
    % the following instructions should take care of the pure multiclass problems. 
    % A pattern is correct if its maximum value is in the correct position
    [vv,ii]=max(dataSet.trainSet.targets(:,trainIndex),[],1);
    [v2,i2]=max(trainOutState.outNetState.outs(:,trainIndex),[],1);
    testing.optimal.trainSet.mistakenPatternIndex=find(ii-i2);
end
evaluateAccuracyOnGraphs('trainSet','optimal')

%% save outputs, useful to evaluate equal error rate
testing.optimal.trainSet.out=trainOutState.outNetState.outs;

testing.optimal.trainSet.mistakenPatterns=dataSet.trainSet.nodeLabels(:,testing.optimal.trainSet.mistakenPatternIndex);
testing.optimal.trainSet.mistakenTargets=dataSet.trainSet.targets(testing.optimal.trainSet.mistakenPatternIndex);
testing.optimal.trainSet.accuracy=1-(size(testing.optimal.trainSet.mistakenPatternIndex(:),1)/supervisedNodesNumberTrain);

if dynamicSystem.config.useValidation

	% evaluating current parameters on validationSet
	if saveMem
	    try
		load('data.mat')
	    catch
		err(0,'Can''t load data.mat in test4classification. Too much memory needed?')
	    end
	    dataSet.trainSet=[];
	    dataSet.testSet=[];
	    pack;
	end

	[x,currentValidationForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
	    zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),'validationSet',0);
	[testing.current.validationSet.error,currentValidationOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,0);

	if size(dataSet.validationSet.targets,1)==1
	    testing.current.validationSet.mistakenPatternIndex=find((dataSet.validationSet.targets>0 & ...
	    currentValidationOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
	    (dataSet.validationSet.targets<0 & currentValidationOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
	    testing.current.validationSet.mistakenPatternIndex=intersect(testing.current.validationSet.mistakenPatternIndex,validationIndex);
	else
	    % the following instructions should take care of the pure multiclass problems. 
	    % A pattern is correct if its maximum value is in the correct position
	    [vv,ii]=max(dataSet.validationSet.targets(:,validationIndex),[],1);
	    [v2,i2]=max(currentValidationOutState.outNetState.outs(:,validationIndex),[],1);
	    testing.current.validationSet.mistakenPatternIndex=find(ii-i2);
	end
	evaluateAccuracyOnGraphs('validationSet','current')

	%% save outputs, useful to evaluate equal error rate
	testing.current.validationSet.out=currentValidationOutState.outNetState.outs;

	testing.current.validationSet.mistakenPatterns=dataSet.validationSet.nodeLabels(:,testing.current.validationSet.mistakenPatternIndex);
	testing.current.validationSet.mistakenTargets=dataSet.validationSet.targets(testing.current.validationSet.mistakenPatternIndex);
	testing.current.validationSet.accuracy=1-(size(testing.current.validationSet.mistakenPatternIndex(:),1)/supervisedNodesNumberValidation);


	% evaluating optimal parameters on validationSet
	[x,validationForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),...
	    'validationSet',1);
	[testing.optimal.validationSet.error,validationOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,1);

	if size(dataSet.validationSet.targets,1)==1
	    testing.optimal.validationSet.mistakenPatternIndex=find((dataSet.validationSet.targets>0 & ...
	    validationOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
	    (dataSet.validationSet.targets<0 & validationOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
	    testing.optimal.validationSet.mistakenPatternIndex=intersect(testing.optimal.validationSet.mistakenPatternIndex,validationIndex);
	else
	    % the following instructions should take care of the pure multiclass problems. 
	    % A pattern is correct if its maximum value is in the correct position
	    [vv,ii]=max(dataSet.validationSet.targets(:,validationIndex),[],1);
	    [v2,i2]=max(validationOutState.outNetState.outs(:,validationIndex),[],1);
	    testing.optimal.validationSet.mistakenPatternIndex=find(ii-i2);
	end
	evaluateAccuracyOnGraphs('validationSet','optimal')

	%% save outputs, useful to evaluate equal error rate
	testing.optimal.validationSet.out=validationOutState.outNetState.outs;
	    
	testing.optimal.validationSet.mistakenPatterns=dataSet.validationSet.nodeLabels(:,testing.optimal.validationSet.mistakenPatternIndex);
	testing.optimal.validationSet.mistakenTargets=dataSet.validationSet.targets(testing.optimal.validationSet.mistakenPatternIndex);
	testing.optimal.validationSet.accuracy=1-(size(testing.optimal.validationSet.mistakenPatternIndex(:),1)/supervisedNodesNumberValidation);
end

% evaluating current parameters on testSet

if saveMem
    try
        load('data.mat')
    catch
        err(0,'Can''t load data.mat in test4classification. Too much memory needed?')
    end
    dataSet.trainSet=[];
    dataSet.validationSet=[];
    pack;
end

[x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
    zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),'testSet',0);
[testing.current.testSet.error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,0);



if size(dataSet.testSet.targets,1)==1
    testing.current.testSet.mistakenPatternIndex=find((dataSet.testSet.targets>0 & ...
    currentTestOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
    (dataSet.testSet.targets<0 & currentTestOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
    testing.current.testSet.mistakenPatternIndex=intersect(testing.current.testSet.mistakenPatternIndex,testIndex);
else
    % the following instructions should take care of the pure multiclass problems. 
    % A pattern is correct if its maximum value is in the correct position
    [vv,ii]=max(dataSet.testSet.targets(:,testIndex),[],1);
    [v2,i2]=max(currentTestOutState.outNetState.outs(:,testIndex),[],1);
    testing.current.testSet.mistakenPatternIndex=find(ii-i2);
end
evaluateAccuracyOnGraphs('testSet','current')

%% save outputs, useful to evaluate equal error rate
testing.current.testSet.out=currentTestOutState.outNetState.outs;

testing.current.testSet.mistakenPatterns=dataSet.testSet.nodeLabels(:,testing.current.testSet.mistakenPatternIndex);
testing.current.testSet.mistakenTargets=dataSet.testSet.targets(testing.current.testSet.mistakenPatternIndex);
testing.current.testSet.accuracy=1-(size(testing.current.testSet.mistakenPatternIndex(:),1)/supervisedNodesNumberTest);


% evaluating optimal parameters on testSet
[x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',1);
[testing.optimal.testSet.error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);

if size(dataSet.testSet.targets,1)==1
    testing.optimal.testSet.mistakenPatternIndex=find((dataSet.testSet.targets>0 & ...
    testOutState.outNetState.outs<dataSet.config.rejectUpperThreshold) | ...
    (dataSet.testSet.targets<0 & testOutState.outNetState.outs>dataSet.config.rejectLowerThreshold));
    testing.optimal.testSet.mistakenPatternIndex=intersect(testing.optimal.testSet.mistakenPatternIndex,testIndex);
else
    % the following instructions should take care of the pure multiclass problems. 
    % A pattern is correct if its maximum value is in the correct position
    [vv,ii]=max(dataSet.testSet.targets(:,testIndex),[],1);
    [v2,i2]=max(testOutState.outNetState.outs(:,testIndex),[],1);
    testing.optimal.testSet.mistakenPatternIndex=find(ii-i2);
end
evaluateAccuracyOnGraphs('testSet','optimal')

%% save outputs, useful to evaluate equal error rate
testing.optimal.testSet.out=testOutState.outNetState.outs;
    
testing.optimal.testSet.mistakenPatterns=dataSet.testSet.nodeLabels(:,testing.optimal.testSet.mistakenPatternIndex);
testing.optimal.testSet.mistakenTargets=dataSet.testSet.targets(testing.optimal.testSet.mistakenPatternIndex);
testing.optimal.testSet.accuracy=1-(size(testing.optimal.testSet.mistakenPatternIndex(:),1)/supervisedNodesNumberTest);

%% creating validationSet out
testing.optimal.validationSet.out=learning.current.optimalValidationOut;



if saveMem
    try
        load('data.mat')
    catch
        err(0,'Can''t load data.mat in test4classification. Too much memory needed?')
    end
end
displayTestRes
catch
    msgstr = lasterr;
    if ~saveMem     % i.e. if this is the first error
        if size(strfind(msgstr,'Out of memory'),1)>0
            warn(0,'There has been an Out of memory error in the test4classification function. Trying again with saveMem=1');
            test4classification(1);
        else
            err(0,msgstr);
            return;
        end
    else
        if size(strfind(msgstr,'Out of memory'),1)>0
            err(0,'Unrecoverable Out of memory error in test4classification. Too much memory needed?')
            message1('Leaving the data in dataerr.mat')
            system('mv data.mat dataerr.mat');
        else
            err(0,msgstr);
            return
        end
    end   
end
%system('rm -f data.mat');
