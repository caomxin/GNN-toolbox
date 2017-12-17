function testMLP(trSet,trTargets,teSet,teTargets)

if nargin==0
    disp('Usage: testMLP(trainSet,trainTargets,testSet,testTargets)')
    return;
end

global MLP MLPLearning MLPTesting


nodeLabelSize=size(teSet,2);


%% trainSet
sizeTrain=size(trSet,2);
h=tanh(MLPLearning.optimalParameters.weights1*trSet+repmat(MLPLearning.optimalParameters.bias1, [1 nodeLabelSize]));
if strcmp(MLP.outFunction,'linear')
    outs=MLPLearning.optimalParameters.weights2*h+repmat(MLPLearning.optimalParameters.bias2, [1 nodeLabelSize]);
else
    outs=tanh(MLPLearning.optimalParameters.weights2*h+repmat(MLPLearning.optimalParameters.bias2, [1 nodeLabelSize]));
end

delta=outs-trTargets;
MLPTesting.trainSet.out=outs;
MLPTesting.trainSet.error=(delta*delta')/2;

MLPTesting.trainSet.mistakenPatternIndex=find((trTargets>0 & outs<0) | (trTargets<0 & outs>0));
MLPTesting.trainSet.mistakenPatterns=trSet(:,MLPTesting.trainSet.mistakenPatternIndex);
MLPTesting.trainSet.mistakenTargets=trTargets(MLPTesting.trainSet.mistakenPatternIndex);
smp=size(MLPTesting.trainSet.mistakenPatternIndex,2);
MLPTesting.trainSet.accuracy=1-(smp/sizeTrain);



%% testSet
sizeTest=size(teSet,2);
h=tanh(MLPLearning.optimalParameters.weights1*teSet+repmat(MLPLearning.optimalParameters.bias1, [1 nodeLabelSize]));
if strcmp(MLP.outFunction,'linear')
    outs=MLPLearning.optimalParameters.weights2*h+repmat(MLPLearning.optimalParameters.bias2, [1 nodeLabelSize]);
else
    outs=tanh(MLPLearning.optimalParameters.weights2*h+repmat(MLPLearning.optimalParameters.bias2, [1 nodeLabelSize]));
end
delta=outs-teTargets;
MLPTesting.testSet.out=outs;
MLPTesting.testSet.error=(delta*delta')/2;

MLPTesting.testSet.mistakenPatternIndex=find((teTargets>0 & outs<0) | (teTargets<0 & outs>0));
MLPTesting.testSet.mistakenPatterns=teSet(:,MLPTesting.testSet.mistakenPatternIndex);
MLPTesting.testSet.mistakenTargets=teTargets(MLPTesting.testSet.mistakenPatternIndex);
smp=size(MLPTesting.testSet.mistakenPatternIndex,2);
MLPTesting.testSet.accuracy=1-(smp/sizeTest);

%display the results
message1('-----------------------------------------------------------------------------');
message1([sprintf('Classification Accuracy on trainSet:\t\t') num2str(MLPTesting.trainSet.accuracy*100) '%'])
message1([sprintf('Train error: \t\t\t\t\t') num2str(MLPTesting.trainSet.error)])
message1('---------------------------------------------');
message1([sprintf('Classification Accuracy on testSet:\t\t') num2str(MLPTesting.testSet.accuracy*100) '%'])
message1([sprintf('Test error: \t\t\t\t\t') num2str(MLPTesting.testSet.error)])
message1('-----------------------------------------------------------------------------');
