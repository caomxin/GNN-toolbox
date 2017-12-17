function normalizeSet
    
    global dataSet

    
    if (isfield(dataSet, 'trainSet'))
        Mn=max(dataSet.trainSet.nodeLabels');
        mn=min(dataSet.trainSet.nodeLabels');
        Mn=Mn-mn;
        Ml=max(dataSet.trainSet.edgeLabels');
        ml=min(dataSet.trainSet.edgeLabels');
        Ml=Ml-ml;
        dataSet.trainSet.nodeLabels=normalize_(dataSet.trainSet.nodeLabels,Mn,mn);
        dataSet.trainSet.edgeLabels=normalize_(dataSet.trainSet.edgeLabels,Ml,ml);
    else
        disp('ERROR: trainSet not found!!!');
        return
    end
    
    if (isfield(dataSet, 'validationSet'))
        dataSet.validationSet.nodeLabels=normalize_(dataSet.validationSet.nodeLabels,Mn,mn);
        dataSet.validationSet.edgeLabels=normalize_(dataSet.validationSet.edgeLabels,Ml,ml);
    end
    if (isfield(dataSet, 'testSet'))
        dataSet.testSet.nodeLabels=normalize_(dataSet.testSet.nodeLabels,Mn,mn);
        dataSet.testSet.edgeLabels=normalize_(dataSet.testSet.edgeLabels,Ml,ml);
    end
    
    
end

function [R]=normalize_(F,M,m)
    features=F';
    for i=1:size(features,2)
        features(:,i) = (features(:,i)-m(i))/M(i);
    end
    R=features';
end