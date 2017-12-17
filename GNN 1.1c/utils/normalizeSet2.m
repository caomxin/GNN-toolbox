function normalizeSet2()

    global dataSet;
    compute('trainSet','nodeLabels');
    compute('trainSet','edgeLabels');
    compute('validationSet','nodeLabels');
    compute('validationSet','edgeLabels');
    compute('testSet','nodeLabels');
    compute('testSet','edgeLabels');
    
end

function compute(set,type)

    global dataSet;
    A=dataSet.(set).(type);
    %disp strcat( set, type, '\n\t Orig size', num2str(size(A))
    M=dataSet.trainSet.mean.(type);
    %disp strcat( '\n\t mean size', num2str(size(M))
    sA=dataSet.trainSet.var.(type);
    %disp strcat( '\n\t sA size', num2str(size(M))
    A0=A-repmat(M, 1, size(A,2)); 
    %disp strcat( '\n\t A0 size', num2str(size(M))
    dataSet.(set).(type) = A0./repmat(sA, 1, size(A,2));
    %disp strcat( '\n\t A0 size', num2str(size(M))
end

