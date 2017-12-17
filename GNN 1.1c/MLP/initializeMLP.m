function initializeMLP(nHidden,outFunction,learningSteps,stepsForValidation,balanceDataset,patternSize)

if nargin==0 || ~isscalar(nHidden) || (~strcmp(outFunction,'linear') && ~strcmp(outFunction,'tanh'))
    disp('Usage: initializeMLP(numHidden,outFunction,learningSteps,stepsForValidation,balanceDataset,patternSize)')
    return;
end

global MLP MLPLearning MLPTesting

%% dataset balancing
MLP.balanceDataset=balanceDataset;

MLP.nHiddens=nHidden;
MLP.outFunction=outFunction;
MLPLearning.learningSteps=learningSteps;
MLPLearning.stepsForValidation=stepsForValidation;

MLP.weightRange=0.01;
MLP.nInputs=patternSize;
MLP.nOuts=1;
MLPLearning.bestErrorOnValidation=realmax;
MLPLearning.history.validationErrorHistory=0;

%% rprop params
MLP.rProp.deltaMax=50;
MLP.rProp.deltaMin=1e-6;
MLP.rProp.etaP=1.2;
MLP.rProp.etaM=0.5;

%% saturation params
MLP.useSaturationControl=1;
MLP.saturationThreshold=0.99;
MLP.saturationCoeff=0.1;

%% initialization
rand('state',sum(100*clock))
MLP.parameters.weights1=2*(rand(MLP.nHiddens,MLP.nInputs)-0.5)*MLP.weightRange;
MLP.parameters.bias1=2*(rand(MLP.nHiddens,1)-0.5)*MLP.weightRange;
MLP.parameters.weights2=2*(rand(MLP.nOuts,MLP.nHiddens)-0.5)*MLP.weightRange;
MLP.parameters.bias2=2*(rand(MLP.nOuts,1)-0.5)*MLP.weightRange;
MLPLearning.learningRate=0.001;
MLPLearning.oldP=MLP.parameters;
MLPLearning.nSteps=1;
for it1=fieldnames(MLP.parameters)'
    MLPLearning.rProp.delta.(char(it1))=0.001*ones(size(MLP.parameters.(char(it1))));
    MLPLearning.rProp.deltaW.(char(it1))=zeros(size(MLP.parameters.(char(it1))));
    MLPLearning.rProp.oldGradient.(char(it1))=zeros(size(MLP.parameters.(char(it1))));
end
