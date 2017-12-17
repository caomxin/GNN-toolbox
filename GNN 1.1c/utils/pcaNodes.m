function pcaNodes(nN, nE)

    global dataSet
    [N cN MN sN]=pcaTrain(dataSet.trainSet.nodeLabels');
    dataSet.trainSet.nodeLabels=(N(:,1:nN))';
    
    [E cE ME sE]=pcaTrain(dataSet.trainSet.edgeLabels');
    dataSet.trainSet.edgeLabels=(E(:,1:nE))';
    
    if (isfield(dataSet, 'validationSet'))
	disp 'OOOOOOOOOOOOOOOOOO'
        doPCA('validationSet', 'nodeLabels', MN, sN, cN, nN);
        doPCA('validationSet', 'edgeLabels', ME, sE, cE, nE);
    end

    dataSet.config.nodeLabelsDim=nN;
    dataSet.config.edgeLabelsDim=nE;
end


function [nodes coefs M sA]=pcaTrain(A)
    M=mean(A);
    A0=A-repmat(M, size(A,1),1); 
    sA=std(A0);
    A0s = A0./repmat(sA, size(A,1),1);    
    [coefs nodes] = princomp(A0s);
end

function doPCA(set, type, M, sA, coefs, n)
    global dataSet
    A=dataSet.(set).(type)';
    A0=A-repmat(M, size(A,1),1); 
    A0s = A0./repmat(sA, size(A,1),1);    
    dataSet.(set).(type)=(A0s*coefs)';
    dataSet.(set).(type)=dataSet.(set).(type)(1:n,:);
end
