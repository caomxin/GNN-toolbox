function testComparisonNet

global dataSet comparisonNet comparisonNetLearning comparisonNetTesting

trainPatterns=size(dataSet.trainSet.nodeLabels,2);
testPatterns=size(dataSet.testSet.nodeLabels,2);

% trainSet optimal
supervisedNodesTrain=find(sum(dataSet.trainSet.maskMatrix>0)+sum(dataSet.trainSet.maskMatrix'>0));
sizeTrain=length(supervisedNodesTrain);
h=tanh(comparisonNetLearning.optimalParameters.weights1*dataSet.trainSet.nodeLabels+repmat(comparisonNetLearning.optimalParameters.bias1, [1 trainPatterns]));
if strcmp(comparisonNet.outFcn,'linear')
    trOuts=comparisonNetLearning.optimalParameters.weights2*h+repmat(comparisonNetLearning.optimalParameters.bias2, [1 trainPatterns]);
else
    trOuts=tanh(comparisonNetLearning.optimalParameters.weights2*h+repmat(comparisonNetLearning.optimalParameters.bias2, [1 trainPatterns]));
end
tmp=trOuts-dataSet.trainSet.targets;
comparisonNetTesting.trainSet.optimal.out=trOuts;
comparisonNetTesting.trainSet.optimal.delta=(dataSet.trainSet.maskMatrix*(tmp'))';
comparisonNetTesting.trainSet.optimal.error=(comparisonNetTesting.trainSet.optimal.delta*tmp')/2;

%testSet optimal
supervisedNodesTest=find(sum(dataSet.testSet.maskMatrix>0)+sum(dataSet.testSet.maskMatrix'>0));
sizeTest=length(supervisedNodesTest);
h=tanh(comparisonNetLearning.optimalParameters.weights1*dataSet.testSet.nodeLabels+repmat(comparisonNetLearning.optimalParameters.bias1, [1 testPatterns]));
if strcmp(comparisonNet.outFcn,'linear')
    teOuts=comparisonNetLearning.optimalParameters.weights2*h+repmat(comparisonNetLearning.optimalParameters.bias2, [1 testPatterns]);
else
    teOuts=tanh(comparisonNetLearning.optimalParameters.weights2*h+repmat(comparisonNetLearning.optimalParameters.bias2, [1 testPatterns]));
end
tmp=teOuts-dataSet.testSet.targets;
comparisonNetTesting.testSet.optimal.out=teOuts;
comparisonNetTesting.testSet.optimal.delta=(dataSet.testSet.maskMatrix*(tmp'))';
comparisonNetTesting.testSet.optimal.error=(comparisonNetTesting.testSet.optimal.delta*tmp')/2;

% trainSet current
h=tanh(comparisonNet.parameters.weights1*dataSet.trainSet.nodeLabels+repmat(comparisonNet.parameters.bias1, [1 trainPatterns]));
if strcmp(comparisonNet.outFcn,'linear')
    trOuts=comparisonNet.parameters.weights2*h+repmat(comparisonNet.parameters.bias2, [1 trainPatterns]);
else
    trOuts=tanh(comparisonNet.parameters.weights2*h+repmat(comparisonNet.parameters.bias2, [1 trainPatterns]));
end
tmp=trOuts-dataSet.trainSet.targets;
comparisonNetTesting.trainSet.current.out=trOuts;
comparisonNetTesting.trainSet.current.delta=(dataSet.trainSet.maskMatrix*(tmp'))';
comparisonNetTesting.trainSet.current.error=(comparisonNetTesting.trainSet.current.delta*tmp')/2;

%testSet current
h=tanh(comparisonNet.parameters.weights1*dataSet.testSet.nodeLabels+repmat(comparisonNet.parameters.bias1, [1 testPatterns]));
if strcmp(comparisonNet.outFcn,'linear')
    teOuts=comparisonNet.parameters.weights2*h+repmat(comparisonNet.parameters.bias2, [1 testPatterns]);
else
    teOuts=tanh(comparisonNet.parameters.weights2*h+repmat(comparisonNet.parameters.bias2, [1 testPatterns]));
end
tmp=teOuts-dataSet.testSet.targets;
comparisonNetTesting.testSet.current.out=teOuts;
comparisonNetTesting.testSet.current.delta=(dataSet.testSet.maskMatrix*(tmp'))';
comparisonNetTesting.testSet.current.error=(comparisonNetTesting.testSet.current.delta*tmp')/2;

if  strcmp(dataSet.config.type,'classification')
    % trainSet optimal
    comparisonNetTesting.trainSet.optimal.mistakenPatternIndex=intersect(find((dataSet.trainSet.targets>0 & comparisonNetTesting.trainSet.optimal.out <dataSet.config.rejectUpperThreshold ) | ...
        (dataSet.trainSet.targets<0 & comparisonNetTesting.trainSet.optimal.out > dataSet.config.rejectLowerThreshold)),supervisedNodesTrain);
    comparisonNetTesting.trainSet.optimal.mistakenPatterns=dataSet.trainSet.nodeLabels(:,comparisonNetTesting.trainSet.optimal.mistakenPatternIndex);
    comparisonNetTesting.trainSet.optimal.mistakenTargets=dataSet.trainSet.targets(comparisonNetTesting.trainSet.optimal.mistakenPatternIndex);
    smpot=size(comparisonNetTesting.trainSet.optimal.mistakenPatternIndex,2);
    comparisonNetTesting.trainSet.optimal.accuracy=1-(smpot/sizeTrain);
    
    % testSet optimal
    comparisonNetTesting.testSet.optimal.mistakenPatternIndex=intersect(find((dataSet.testSet.targets>0 & comparisonNetTesting.testSet.optimal.out <dataSet.config.rejectUpperThreshold) | ...
        (dataSet.testSet.targets<0 & comparisonNetTesting.testSet.optimal.out > dataSet.config.rejectLowerThreshold) ),supervisedNodesTest);
    comparisonNetTesting.testSet.optimal.mistakenPatterns=dataSet.testSet.nodeLabels(:,comparisonNetTesting.testSet.optimal.mistakenPatternIndex);
    comparisonNetTesting.testSet.optimal.mistakenTargets=dataSet.testSet.targets(comparisonNetTesting.testSet.optimal.mistakenPatternIndex);
    smp=size(comparisonNetTesting.testSet.optimal.mistakenPatternIndex,2);
    comparisonNetTesting.testSet.optimal.accuracy=1-(smp/sizeTest);

    % evaluate accuracy on graphs (instead of on nodes)
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        % it makes sense to evaluate accuracy on graphs
        a={'trainSet','testSet'};
        for ai=1:size(a,2)
            eval(['err=zeros(size(dataSet.' a{ai} '.targets));']);
            eval(['err(comparisonNetTesting.'  a{ai} '.optimal.mistakenPatternIndex)=1;']);
            eval(['g_err=reshape(err,dataSet.config.graphDim,dataSet.' a{ai} '.graphNum);']);
            eval(['comparisonNetTesting.' a{ai} '.accuracyOnGraphs=size(find(sum(g_err)==0),2)/dataSet.' a{ai} '.graphNum;']);
        end
    end
    
        % trainSet current
    comparisonNetTesting.trainSet.current.mistakenPatternIndex=intersect(find((dataSet.trainSet.targets>0 & comparisonNetTesting.trainSet.current.out <dataSet.config.rejectUpperThreshold ) | ...
        (dataSet.trainSet.targets<0 & comparisonNetTesting.trainSet.current.out > dataSet.config.rejectLowerThreshold)),supervisedNodesTrain);
    comparisonNetTesting.trainSet.current.mistakenPatterns=dataSet.trainSet.nodeLabels(:,comparisonNetTesting.trainSet.current.mistakenPatternIndex);
    comparisonNetTesting.trainSet.current.mistakenTargets=dataSet.trainSet.targets(comparisonNetTesting.trainSet.current.mistakenPatternIndex);
    smpot=size(comparisonNetTesting.trainSet.current.mistakenPatternIndex,2);
    comparisonNetTesting.trainSet.current.accuracy=1-(smpot/sizeTrain);
    
    % testSet current
    comparisonNetTesting.testSet.current.mistakenPatternIndex=intersect(find((dataSet.testSet.targets>0 & comparisonNetTesting.testSet.current.out <dataSet.config.rejectUpperThreshold) | ...
        (dataSet.testSet.targets<0 & comparisonNetTesting.testSet.current.out > dataSet.config.rejectLowerThreshold) ),supervisedNodesTest);
    comparisonNetTesting.testSet.current.mistakenPatterns=dataSet.testSet.nodeLabels(:,comparisonNetTesting.testSet.current.mistakenPatternIndex);
    comparisonNetTesting.testSet.current.mistakenTargets=dataSet.testSet.targets(comparisonNetTesting.testSet.current.mistakenPatternIndex);
    smp=size(comparisonNetTesting.testSet.current.mistakenPatternIndex,2);
    comparisonNetTesting.testSet.current.accuracy=1-(smp/sizeTest);

    % evaluate accuracy on graphs (instead of on nodes)
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        % it makes sense to evaluate accuracy on graphs
        a={'trainSet','testSet'};
        for ai=1:size(a,2)
            eval(['err=zeros(size(dataSet.' a{ai} '.targets));']);
            eval(['err(comparisonNetTesting.'  a{ai} '.current.mistakenPatternIndex)=1;']);
            eval(['g_err=reshape(err,dataSet.config.graphDim,dataSet.' a{ai} '.graphNum);']);
            eval(['comparisonNetTesting.' a{ai} '.accuracyOnGraphs=size(find(sum(g_err)==0),2)/dataSet.' a{ai} '.graphNum;']);
        end
    end


    %display the results
    message1('-----------------------------------------------------------------------------');
    message1('OPTIMAL');
    message1([sprintf('Classification Accuracy on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.optimal.accuracy*100) '%'])
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        message1([sprintf('Classification AccuracyOnGraphs on trainSet:\t') num2str(comparisonNetTesting.trainSet.optimal.accuracyOnGraphs*100) '%'])
    end
    message1([sprintf('Train error: \t\t\t\t\t') num2str(comparisonNetTesting.trainSet.optimal.error)])
    message1(' ');
   message1([sprintf('Classification Accuracy on testSet:\t\t') num2str(comparisonNetTesting.testSet.optimal.accuracy*100) '%'])
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        message1([sprintf('Classification AccuracyOnGraphs on testSet:\t') num2str(comparisonNetTesting.testSet.optimal.accuracyOnGraphs*100) '%'])
    end
    message1([sprintf('Test error: \t\t\t\t\t') num2str(comparisonNetTesting.testSet.optimal.error)])
    message1('-----------------------------------------------------------------------------');
    message1('CURRENT');
    message1([sprintf('Classification Accuracy on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.current.accuracy*100) '%'])
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        message1([sprintf('Classification AccuracyOnGraphs on trainSet:\t') num2str(comparisonNetTesting.trainSet.current.accuracyOnGraphs*100) '%'])
    end
    message1([sprintf('Train error: \t\t\t\t\t') num2str(comparisonNetTesting.trainSet.current.error)])
      message1(' ');
    message1([sprintf('Classification Accuracy on testSet:\t\t') num2str(comparisonNetTesting.testSet.current.accuracy*100) '%'])
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        message1([sprintf('Classification AccuracyOnGraphs on testSet:\t') num2str(comparisonNetTesting.testSet.current.accuracyOnGraphs*100) '%'])
    end
    message1([sprintf('Test error: \t\t\t\t\t') num2str(comparisonNetTesting.testSet.current.error)])
    message1('-----------------------------------------------------------------------------');
else
    %trainSet optimal
    comparisonNetTesting.trainSet.optimal.maxRelativeError=max(abs(comparisonNetTesting.trainSet.optimal.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain)));
    comparisonNetTesting.trainSet.optimal.acc5percent=size(find(abs(comparisonNetTesting.trainSet.optimal.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain))<0.05),2)/sizeTrain;
    comparisonNetTesting.trainSet.optimal.acc10percent=size(find(abs(comparisonNetTesting.trainSet.optimal.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain))<0.1),2)/sizeTrain;
    comparisonNetTesting.trainSet.optimal.maxError=max(abs(comparisonNetTesting.trainSet.optimal.delta(supervisedNodesTrain)));

    %testSet optimal
    comparisonNetTesting.testSet.optimal.maxRelativeError=max(abs(comparisonNetTesting.testSet.optimal.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest)));
    comparisonNetTesting.testSet.optimal.acc5percent=size(find(abs(comparisonNetTesting.testSet.optimal.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest))<0.05),2)/sizeTest;
    comparisonNetTesting.testSet.optimal.acc10percent=size(find(abs(comparisonNetTesting.testSet.optimal.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest))<0.1),2)/sizeTest;
    comparisonNetTesting.testSet.optimal.maxError=max(abs(comparisonNetTesting.testSet.optimal.delta(supervisedNodesTest)));

    %trainSet current
    comparisonNetTesting.trainSet.current.maxRelativeError=max(abs(comparisonNetTesting.trainSet.current.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain)));
    comparisonNetTesting.trainSet.current.acc5percent=size(find(abs(comparisonNetTesting.trainSet.current.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain))<0.05),2)/sizeTrain;
    comparisonNetTesting.trainSet.current.acc10percent=size(find(abs(comparisonNetTesting.trainSet.current.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain))<0.1),2)/sizeTrain;
    comparisonNetTesting.trainSet.current.maxError=max(abs(comparisonNetTesting.trainSet.current.delta(supervisedNodesTrain)));

    %testSet current
    comparisonNetTesting.testSet.current.maxRelativeError=max(abs(comparisonNetTesting.testSet.current.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest)));
    comparisonNetTesting.testSet.current.acc5percent=size(find(abs(comparisonNetTesting.testSet.current.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest))<0.05),2)/sizeTest;
    comparisonNetTesting.testSet.current.acc10percent=size(find(abs(comparisonNetTesting.testSet.current.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest))<0.1),2)/sizeTest;
    comparisonNetTesting.testSet.current.maxError=max(abs(comparisonNetTesting.testSet.current.delta(supervisedNodesTest)));

    
    %display the results
    message1(sprintf('\n----------------------------------------'));
    message1(sprintf('OPTIMAL'));
    message1([sprintf('Error on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.optimal.error)])
    message1([sprintf('maxError on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.optimal.maxError)])
    message1([sprintf('maxRelativeError on trainSet:\t') num2str(comparisonNetTesting.trainSet.optimal.maxRelativeError)])
    message1([sprintf('err < 0.05:\t\t\t') num2str(comparisonNetTesting.trainSet.optimal.acc5percent*100,'%4.2f') '%'])
    message1([sprintf('err < 0.1:\t\t\t') num2str(comparisonNetTesting.trainSet.optimal.acc10percent*100,'%4.2f') '%'])
    message1(' ');
    message1([sprintf('Error on testSet:\t\t') num2str(comparisonNetTesting.testSet.optimal.error)])
    message1([sprintf('maxError on testSet:\t\t') num2str(comparisonNetTesting.testSet.optimal.maxError)])
    message1([sprintf('maxRelativeError on testSet:\t') num2str(comparisonNetTesting.testSet.optimal.maxRelativeError)])
    message1([sprintf('err < 0.05:\t\t\t') num2str(comparisonNetTesting.testSet.optimal.acc5percent*100,'%4.2f') '%'])
    message1([sprintf('err < 0.1:\t\t\t') num2str(comparisonNetTesting.testSet.optimal.acc10percent*100,'%4.2f') '%'])
    message1(sprintf('\n----------------------------------------'));
    message1(sprintf('CURRENT'));
    message1([sprintf('Error on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.current.error)])
    message1([sprintf('maxError on trainSet:\t\t') num2str(comparisonNetTesting.trainSet.current.maxError)])
    message1([sprintf('maxRelativeError on trainSet:\t') num2str(comparisonNetTesting.trainSet.current.maxRelativeError)])
    message1([sprintf('err < 0.05:\t\t\t') num2str(comparisonNetTesting.trainSet.current.acc5percent*100,'%4.2f') '%'])
    message1([sprintf('err < 0.1:\t\t\t') num2str(comparisonNetTesting.trainSet.current.acc10percent*100,'%4.2f') '%'])
    message1(' ');
    message1([sprintf('Error on testSet:\t\t') num2str(comparisonNetTesting.testSet.current.error)])
    message1([sprintf('maxError on testSet:\t\t') num2str(comparisonNetTesting.testSet.current.maxError)])
    message1([sprintf('maxRelativeError on testSet:\t') num2str(comparisonNetTesting.testSet.current.maxRelativeError)])
    message1([sprintf('err < 0.05:\t\t\t') num2str(comparisonNetTesting.testSet.current.acc5percent*100,'%4.2f') '%'])
    message1([sprintf('err < 0.1:\t\t\t') num2str(comparisonNetTesting.testSet.current.acc10percent*100,'%4.2f') '%'])
    message1(sprintf('----------------------------------------\n'));
end