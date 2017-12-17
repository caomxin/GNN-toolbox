function initializeComparisonNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% activation functions must be chosen in the learnComparisonNet.m and in
% testComparisonNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global dataSet comparisonNet comparisonNetLearning

comparisonNet=[];
comparisonNetLearning=[];

comparisonNet.useSaturationControl=1;


comparisonNet.nHiddens=20;
comparisonNetLearning.learningSteps=2000;


comparisonNet.weightRange=0.01;
comparisonNet.nInputs=size(dataSet.trainSet.nodeLabels,1);
comparisonNet.nOuts=1;


rand('state',sum(100*clock))

comparisonNet.parameters.weights1=2*(rand(comparisonNet.nHiddens,comparisonNet.nInputs)-0.5)*comparisonNet.weightRange;
comparisonNet.parameters.bias1=2*(rand(comparisonNet.nHiddens,1)-0.5)*comparisonNet.weightRange;
comparisonNet.parameters.weights2=2*(rand(comparisonNet.nOuts,comparisonNet.nHiddens)-0.5)*comparisonNet.weightRange;
comparisonNet.parameters.bias2=2*(rand(comparisonNet.nOuts,1)-0.5)*comparisonNet.weightRange;
comparisonNetLearning.learningRate=0.001;


comparisonNetLearning.nSteps=1;
comparisonNetLearning.allSteps=1;


comparisonNetLearning.stepsForValidation=100;
comparisonNetLearning.bestErrorOnValidation=realmax;
comparisonNetLearning.history.validationErrorHistory=0;

% rprop initialization
comparisonNet.rProp.deltaMax=50;
comparisonNet.rProp.deltaMin=1e-6;
comparisonNet.rProp.etaP=1.2;
comparisonNet.rProp.etaM=0.5;

%saturation
comparisonNet.saturationThreshold=0.99;
comparisonNet.saturationCoeff=0.1;

for it1=fieldnames(comparisonNet.parameters)'
    comparisonNetLearning.rProp.delta.(char(it1))=0.001*ones(size(comparisonNet.parameters.(char(it1))));
    comparisonNetLearning.rProp.deltaW.(char(it1))=zeros(size(comparisonNet.parameters.(char(it1))));
    comparisonNetLearning.rProp.oldGradient.(char(it1))=zeros(size(comparisonNet.parameters.(char(it1))));
end


comparisonNetLearning.oldP=comparisonNet.parameters;
