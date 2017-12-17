function trainMLP(trSet,trTargets,valSet,valTargets)

if nargin==0 
    disp('Usage: trainMLP(trainSet,trainTargets,validationSet,validationTargets)')
    return;
end
useValidation=1;
if nargin==2
    useValidation=0;
end

global MLP MLPLearning MLPTesting

%% training
MLPLearning.iStart=MLPLearning.nSteps;
trainPatterns=size(trSet,2);
valPatterns=size(valSet,2);

if MLP.balanceDataset
	%% balance dataset rescoring errors on positive examples according to positive/negative ratio
    trPos=find(trTargets==1);
    trNumPos=size(trPos,2);
    trNumNeg=size(find(trTargets==-1),2);
    trPositiveFactor=sqrt(trNumNeg/trNumPos); %% this the multiplication factor for errors on positive patterns
    trMask=ones(size(trTargets));
    trMask(trPos)=trPositiveFactor;   
    if useValidation
        valPos=find(valTargets==1);
        valNumPos=size(valPos,2);
        valNumNeg=size(find(valTargets==-1),2);
        valPositiveFactor=sqrt(valNumNeg/valNumPos); %% this the multiplication factor for errors on positive patterns
        valMask=ones(size(valTargets));
        valMask(valPos)=valPositiveFactor;
    end
end

while MLPLearning.nSteps<=MLPLearning.iStart+MLPLearning.learningSteps
    h=tanh(MLP.parameters.weights1*trSet+repmat(MLP.parameters.bias1, [1 trainPatterns]));
    if strcmp(MLP.outFunction,'linear')
        outs=MLP.parameters.weights2*h+repmat(MLP.parameters.bias2, [1 trainPatterns]);
    else
        outs=tanh(MLP.parameters.weights2*h+repmat(MLP.parameters.bias2, [1 trainPatterns]));
    end
    
    delta=outs-trTargets;
    if MLP.balanceDataset
        delta=delta.*trMask;
    end
    error=delta*delta'/2;

    MLPLearning.history.trainErrorHistory(MLPLearning.nSteps)=error;
    MLPLearning.oldP=MLP.parameters;

    if strcmp(MLP.outFunction,'linear')
        dnet2=delta;
    else
        dnet2=delta .* (1-outs.*outs);
    end
    if MLP.useSaturationControl && ~strcmp(MLP.outFunction,'linear')
        absval=abs(outs)-MLP.saturationThreshold;
        absval(absval<0)=0;
        dnet2 = dnet2 + MLP.saturationCoeff.*absval.*sign(outs);
    end
    gradient.weights2=dnet2*h';
    gradient.bias2=sum(dnet2,2);

    if strcmp(MLP.outFunction,'linear')
        dnet1=(MLP.parameters.weights2'*dnet2);
    else
        dnet1=(MLP.parameters.weights2'*dnet2) .* (1- h.*h);
    end

    if MLP.useSaturationControl
        absval=abs(h)-MLP.saturationThreshold;
        absval(absval<0)=0;
        dnet1 = dnet1 + MLP.saturationCoeff.*absval.*sign(h);
    end

    gradient.weights1=dnet1*trSet';
    gradient.bias1=sum(dnet1,2);

    if mod(MLPLearning.nSteps, MLPLearning.stepsForValidation)==0
        disp(['step: ' num2str(MLPLearning.nSteps) sprintf('\tMLP training error: ') num2str(error)]);
        if useValidation
            h=tanh(MLP.parameters.weights1*valSet+repmat(MLP.parameters.bias1, [1 valPatterns]));
            outs=tanh(MLP.parameters.weights2*h+repmat(MLP.parameters.bias2, [1 valPatterns]));
            vDelta=outs-valTargets;
            if MLP.balanceDataset
                vDelta=vDelta.*valMask;
            end
            MLPLearning.vError=vDelta*vDelta'/2;
            MLPLearning.history.validationErrorHistory=[MLPLearning.history.validationErrorHistory,MLPLearning.vError];
            if (MLPLearning.vError < MLPLearning.bestErrorOnValidation)
                MLPLearning.bestErrorOnValidation=MLPLearning.vError;
                MLPLearning.optimalParameters=MLP.parameters;
                disp([sprintf('\t\t\t') 'best validation error: ' num2str(MLPLearning.vError)])
            end
        else
            MLPLearning.optimalParameters=MLP.parameters;
        end
    end

    %% parameters updating
    for it1=fieldnames(gradient)'
        old4new=gradient.(char(it1)) .* MLPLearning.rProp.oldGradient.(char(it1));
        MLPLearning.rProp.delta.(char(it1)) =...
            (old4new>0) .* min(MLP.rProp.deltaMax,MLP.rProp.etaP * MLPLearning.rProp.delta.(char(it1))) ...
            +(old4new<0) .* max(MLP.rProp.deltaMin,MLP.rProp.etaM * MLPLearning.rProp.delta.(char(it1)))...
            +(old4new==0) .* MLPLearning.rProp.delta.(char(it1));

        MLPLearning.rProp.deltaW.(char(it1)) =...
            (old4new>0) .*(-sign(gradient.(char(it1))) .* MLPLearning.rProp.delta.(char(it1)))...
            +(old4new<0) .* MLPLearning.rProp.deltaW.(char(it1)) ...
            +(old4new==0) .* (-sign(gradient.(char(it1))) .* MLPLearning.rProp.delta.(char(it1)));

        MLP.parameters.(char(it1))=MLPLearning.oldP.(char(it1)) ...
            +(old4new>0) .* MLPLearning.rProp.deltaW.(char(it1)) ...
            -(old4new<0) .* MLPLearning.rProp.deltaW.(char(it1)) ...
            +(old4new==0) .* MLPLearning.rProp.deltaW.(char(it1));

        MLPLearning.rProp.oldGradient.(char(it1))=...
            (old4new>=0) .* gradient.(char(it1))...
            +(old4new<0) .* zeros(size(gradient.(char(it1))));
    end
    MLPLearning.nSteps=MLPLearning.nSteps+1;
end
