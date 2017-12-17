function displayTestRes
% displays results
global testing dataSet learning VisualMode
if VisualMode == 1
    if isfield(testing.current.trainSet, 'accuracyOnGraphs')
        TestFigH=DisplayTestC(testing.current.trainSet.error,testing.current.trainSet.accuracy,...
            testing.optimal.trainSet.error,testing.optimal.trainSet.accuracy,...
            testing.current.testSet.error,testing.current.testSet.accuracy,...
            testing.optimal.testSet.error,testing.optimal.testSet.accuracy,...
            testing.current.trainSet.accuracyOnGraphs, testing.optimal.trainSet.accuracyOnGraphs,...
            testing.current.testSet.accuracyOnGraphs, testing.optimal.testSet.accuracyOnGraphs);
    else
        TestFigH=DisplayTestC(testing.current.trainSet.error, testing.current.trainSet.accuracy,...
            testing.optimal.trainSet.error, testing.optimal.trainSet.accuracy,...
            testing.current.testSet.error,testing.current.testSet.accuracy,...
            testing.optimal.testSet.error,testing.optimal.testSet.accuracy);
    end
    
else
    header='';
    line='';
    accuracies.optimal='';
    accuracies.current='';
    errors=accuracies;

    fmtH='%s%20s';
    fmtA='%s%19.2f%%';
    fmtE='%s%20.5f';
    for set={'trainSet', 'validationSet', 'testSet'}
        if isfield(dataSet, set)
            header=sprintf(fmtH, header, set{:});
            accuracies.optimal=sprintf(fmtA,accuracies.optimal,testing.optimal.(set{:}).accuracy*100);
            accuracies.current=sprintf(fmtA,accuracies.current,testing.current.(set{:}).accuracy*100);
            errors.optimal=sprintf(fmtE,errors.optimal,testing.optimal.(set{:}).error);
            errors.current=sprintf(fmtE,errors.current,testing.current.(set{:}).error);
            line(end+1:end+20)='-'; 
        end
    end

    %message1(sprintf('\n\t\t\tTESTSET\t\t\t\t\t\tTRAINSET'));
    message1(['                 |' header]);
    message1(['-----------------|' line]);
    message1(['OPTIMAL ACCURACY |' accuracies.optimal ]);
    message1(['OPTIMAL ERROR    |' errors.optimal     ]);
    message1( '                 |' );
    message1(['CURRENT ACCURACY |' accuracies.current ]);
    message1(['CURRENT ERROR    |' errors.current     ]);
    message1(['-----------------|' line]);
end
