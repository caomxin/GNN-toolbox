function computeNormalizationFactor()

    global dataSet;
    dataSet.trainSet.mean={};
    dataSet.trainSet.var={};
    compute('nodeLabels');
    compute('edgeLabels');
end


function compute(type)

    global dataSet;
    A=dataSet.trainSet.(type);
    M=mean(A,2);
    A0=A-repmat(M, 1, size(A,2)); 
    sA=std(A0,0,2);
    dataSet.trainSet.mean.(type)=M;
    dataSet.trainSet.var.(type)=sA;

end

