function test4autoassociator
% testing function for classification problems (private)

global dataSet dynamicSystem learning testing TestFigH diffcur diffop

dataSet.testSet.forwardSteps=30;

supervisedNodesNumberTrain=size(find(diag(dataSet.trainSet.maskMatrix)),1);
supervisedNodesNumberTest=size(find(diag(dataSet.testSet.maskMatrix)),1);

if dynamicSystem.config.useValidation
    %cerco la soglia migliore sul validation Set con parametri ottimi
    learning.current.validationState=feval(dynamicSystem.config.forwardFunction,learning.config.maxStepsForValidation,learning.current.validationState,...
        'validationSet',1);

    [learning.current.validationError learning.current.validationOut]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',[],1);

    in=[learning.current.validationState; dataSet.validationSet.nodeLabels];
    Threshold=0;
    corrtmp=0;
    f1tmp=0;
   
    %for Threshold=0.1:dataSet.config.ThresholdPass/2:dataSet.config.ThresholdUp
    for Threshold=0.1:0.1:5
        f1=0;
        corretti=0;
        corrPos=0;
        corrNeg=0;       
        learning.current.validationMistakenPatternIndex=[];
      for i=1:size(learning.current.validationOut.outNetState.outs,2)
            diff= (learning.current.optimalValidationOut(:,i)-in(:,i))'*(learning.current.optimalValidationOut(:,i) - in(:,i));
            t=dataSet.validationSet.targets(i);
            if(diff<Threshold)&&(t>0) corrPos=corrPos+1; end
            if(diff>Threshold)&&(t<0) corrNeg=corrNeg+1; end
            %  corretti=corrPos+(corrNeg*dynamicSystem.config.validationBalancingFact);

            if(diff<=Threshold)&(t>0)|(diff>Threshold)&(t<0)
                corretti=corretti+1;
            else
                learning.current.validationMistakenPatternIndex=[learning.current.validationMistakenPatternIndex,i];
                %if (t>0) errpos=errpos+1;
                %else errneg=errneg+1;
                %end
            end
        end
        %tarCur=mistaken2all(learning.current.validationMistakenPatternIndex,dataSet.validationSet.targets);
        tarCur=dataSet.validationSet.targets;
        tarCur(learning.current.validationMistakenPatternIndex)=-tarCur(learning.current.validationMistakenPatternIndex);
        quality=QualityParameters(dataSet.validationSet.targets,tarCur);
        f1= quality.f1;
        disp(sprintf('Threshold=%f f1=%f',Threshold,f1));
        %f1=corrPos*corrNeg;
        if f1>f1tmp
            f1tmp=f1;
            corrtmp=corretti;
            thretmp=Threshold;
            patttmp=learning.current.validationMistakenPatternIndex;
        end
    end
    learning.current.validationMistakenPatternIndex=patttmp;
    Threshold=thretmp;
    corretti=corrtmp;
    f1=f1tmp;
    disp(sprintf('Best threshold=%f, corretti=%d/%d f1=%f',Threshold,corretti,size(dataSet.testSet.targets,2),f1));


%	Threshold=10.0


    testing.threshold=Threshold;
else
    err(0,'I need validationSet to find optimal autoassociator classification threshold');
    return;
end
 
 
 %valuto il train current con la soglia trovata
 [x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',0);
 [testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);
 in=[x;dataSet.trainSet.nodeLabels];  %GABRIELE
 
 corretti=0;
 
 testing.current.trainSet.mistakenPatternIndex=[];
 for i=1:size(currentTrainOutState.outNetState.outs,2)
     diff= ( currentTrainOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTrainOutState.outNetState.outs(:,i) - in(:,i));
     t=dataSet.trainSet.targets(i);
     if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
         corretti=corretti+1;
         % testing.current.trainSet.targets(i)=1;
     else
         testing.current.trainSet.mistakenPatternIndex=[testing.current.trainSet.mistakenPatternIndex,i];
         % testing.current.trainSet.targets(i)=-1;
     end
 end
 tar=mistaken2all(testing.current.trainSet.mistakenPatternIndex,dataSet.trainSet.targets);
 testing.current.trainSet.prediction=tar;
 
 %% save outputs, useful to evaluate equal error rate
 testing.current.trainSet.out=currentTrainOutState.outNetState.outs;
 testing.current.trainSet.accuracy=corretti/size(currentTrainOutState.outNetState.outs,2);
 
 
 % evaluating optimal parameters on trainSet
 [x,trainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',1);
 [testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,1);
 in=[x;dataSet.trainSet.nodeLabels];
 
 corretti=0;
 testing.optimal.trainSet.mistakenPatternIndex=[];
 
 for i=1:size(trainOutState.outNetState.outs,2)
     diff= ( trainOutState.outNetState.outs(:,i)-in(:,i) )'*( trainOutState.outNetState.outs(:,i) - in(:,i));
     t=dataSet.trainSet.targets(i);
 
     if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
         corretti=corretti+1;
         %testing.optimal.trainSet.targets(i)=1;
     else
         testing.optimal.trainSet.mistakenPatternIndex=[testing.optimal.trainSet.mistakenPatternIndex,i];
         %testing.optimal.trainSet.targets(i)=-1;
     end
 
 end
 tar=mistaken2all(testing.optimal.trainSet.mistakenPatternIndex,dataSet.trainSet.targets);
 testing.optimal.trainSet.prediction=tar;
 
 %% save outputs, useful to evaluate equal error rate
 testing.optimal.trainSet.out=trainOutState.outNetState.outs;
 testing.optimal.trainSet.accuracy=corretti/size(trainOutState.outNetState.outs,2);
 
 
 %valuto il val current con la soglia trovata
 [x,currentValForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),'validationSet',0);
 [testing.current.validationSet.error,currentValOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,0);
 in=[x;dataSet.validationSet.nodeLabels];  %GABRIELE
 
 corretti=0;
 
 testing.current.validationSet.mistakenPatternIndex=[];
 for i=1:size(currentValOutState.outNetState.outs,2)
     diff= (currentValOutState.outNetState.outs(:,i)-in(:,i) )'*( currentValOutState.outNetState.outs(:,i) - in(:,i));
     t=dataSet.validationSet.targets(i);
 
     if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
         corretti=corretti+1;
         % testing.current.trainSet.targets(i)=1;
     else
         testing.current.validationSet.mistakenPatternIndex=[testing.current.validationSet.mistakenPatternIndex,i];
         % testing.current.trainSet.targets(i)=-1;
     end
 end
 tar=mistaken2all(testing.current.validationSet.mistakenPatternIndex,dataSet.validationSet.targets);
 testing.current.validationSet.prediction=tar;
 
 %% save outputs, useful to evaluate equal error rate
 testing.current.validationSet.out=currentValOutState.outNetState.outs;
 testing.current.validationSet.accuracy=corretti/size(currentValOutState.outNetState.outs,2);
 
 % evaluating optimal parameters on validationSet
 [x,valForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps, zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),'validationSet',1);
 [testing.optimal.validationSet.error,valOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,1);
 in=[x;dataSet.validationSet.nodeLabels];
 
 corretti=0;
 testing.optimal.validationSet.mistakenPatternIndex=[];
 
 for i=1:size(valOutState.outNetState.outs,2)
     diff= ( valOutState.outNetState.outs(:,i)-in(:,i) )'*( valOutState.outNetState.outs(:,i) - in(:,i));
     t=dataSet.validationSet.targets(i);   
     if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
         corretti=corretti+1;
         %testing.optimal.trainSet.targets(i)=1;
     else
        testing.optimal.validationSet.mistakenPatternIndex=[testing.optimal.validationSet.mistakenPatternIndex,i];
         %testing.optimal.trainSet.targets(i)=-1;
     end
 
 end
 tar=mistaken2all(testing.optimal.validationSet.mistakenPatternIndex,dataSet.validationSet.targets);
 testing.optimal.validationSet.prediction=tar;
 
 %% save outputs, useful to evaluate equal error rate
 testing.optimal.validationSet.out=valOutState.outNetState.outs;
 
 
 testing.optimal.validationSet.accuracy=corretti/size(valOutState.outNetState.outs,2);


% evaluating current parameters on testSet
%% in general we may not have the targets, so also the prediction is evaluated
%% Moreover the testSet is split into parts with the same number of graphs
global globalTestSet
globalTestSet=dataSet.testSet;
graphsEachTime=12;
startedge=1;
corretti=0;
correttiOpt=0;
cumNodes=cumsum(globalTestSet.nodesPerGraph);
testing.current.testSet.mistakenPatternIndex=[];
testing.current.testSet.out=[];
testing.current.testSet.prediction=[];
testing.current.testSet.error=0;

testing.optimal.testSet.mistakenPatternIndex=[];
testing.optimal.testSet.out=[];
testing.optimal.testSet.prediction=[];
testing.optimal.testSet.error=0;



for piece=1:fix(globalTestSet.graphNum/graphsEachTime)+1
    %% current
    if piece~=fix(globalTestSet.graphNum/graphsEachTime)+1
        disp(sprintf('testSet/current. Testing graphs: %d-%d on %d',graphsEachTime*(piece-1),graphsEachTime*piece,globalTestSet.graphNum));
    else
        disp(sprintf('testSet/current. Testing graphs: %d-%d on %d',graphsEachTime*(piece-1),globalTestSet.graphNum,globalTestSet.graphNum));
    end
    dataSet.testSet=[];
    dataSet.testSet.forwardSteps=globalTestSet.forwardSteps;
    if piece==1
        startnode=1;
    else
        startnode=cumNodes(graphsEachTime*(piece-1))+1;
    end
    if piece==fix(globalTestSet.graphNum/graphsEachTime)+1
        endnode=cumNodes(end);
    else
        endnode=cumNodes(graphsEachTime*piece);
    end
    dataSet.testSet.connMatrix=globalTestSet.connMatrix(startnode:endnode,startnode:endnode);
    dataSet.testSet.maskMatrix=globalTestSet.maskMatrix(startnode:endnode,startnode:endnode);
    dataSet.testSet.nodeLabels=globalTestSet.nodeLabels(:,startnode:endnode);
    dataSet.testSet.targets=globalTestSet.targets(:,startnode:endnode);
    if isfield(globalTestSet,'edgeLabels')
        totedges=sum(sum(dataSet.testSet.connMatrix));
%         size(globalTestSet.edgeLabels)
         startedge
         totedges
        dataSet.testSet.edgeLabels=globalTestSet.edgeLabels(:,startedge:startedge+totedges-1);
    	startedge=startedge+totedges;
    end
    dataSet.testSet.nNodes=endnode-startnode+1;
    dataSet.testSet.graphNum=graphsEachTime;
    prepareDataset('testSet');


    [x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
        zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),'testSet',0);
    [error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,0);
    testing.current.testSet.error=testing.current.testSet.error+error;
    in=[x;dataSet.testSet.nodeLabels];
    tar=[];
    for i=1:size(currentTestOutState.outNetState.outs,2)
        diff= ( currentTestOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTestOutState.outNetState.outs(:,i) - in(:,i));
        t=dataSet.testSet.targets(i);
        if(diff<=Threshold)
            d=1;
            if t>0
                corretti=corretti+1;
            else
                testing.current.testSet.mistakenPatternIndex=[testing.current.testSet.mistakenPatternIndex,i+startnode];
            end
        else
            d=-1;
            if t<0
                corretti=corretti+1;
            else
                testing.current.testSet.mistakenPatternIndex=[testing.current.testSet.mistakenPatternIndex,i+startnode];
            end
        end
        tar=[tar,d];
    end

    testing.current.testSet.prediction=[testing.current.testSet.prediction,tar];
    %% save outputs, useful to evaluate equal error rate
    testing.current.testSet.out=[testing.current.testSet.out,currentTestOutState.outNetState.outs];
    

    %% optimal
    if piece~=fix(globalTestSet.graphNum/graphsEachTime)+1
        disp(sprintf('testSet/optimal. Testing graphs: %d-%d on %d',graphsEachTime*(piece-1),graphsEachTime*piece,globalTestSet.graphNum));
    else
        disp(sprintf('testSet/optimal. Testing graphs: %d-%d on %d',graphsEachTime*(piece-1),globalTestSet.graphNum,globalTestSet.graphNum));
    end
    [x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
        'testSet',1);
    [error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);
    testing.optimal.testSet.error=testing.optimal.testSet.error+error;
    in=[x;dataSet.testSet.nodeLabels];
    tar=[];
    for i=1:size(testOutState.outNetState.outs,2)
        diff= (testOutState.outNetState.outs(:,i)-in(:,i) )'*( testOutState.outNetState.outs(:,i) - in(:,i));
        t=dataSet.testSet.targets(i);
        if(diff<=Threshold)
            d=1;
            if t>0
                correttiOpt=correttiOpt+1;
            else
                testing.optimal.testSet.mistakenPatternIndex=[testing.optimal.testSet.mistakenPatternIndex,i+startnode];
            end
        else
            d=-1;
            if t<0
                correttiOpt=correttiOpt+1;
            else
                testing.optimal.testSet.mistakenPatternIndex=[testing.optimal.testSet.mistakenPatternIndex,i+startnode];
            end
        end
        tar=[tar,d];

    end
    testing.optimal.testSet.prediction=[testing.optimal.testSet.prediction,tar];
    %% save outputs, useful to evaluate equal error rate
    testing.optimal.testSet.out=[testing.optimal.testSet.out,testOutState.outNetState.outs];


end
testing.current.testSet.accuracy=corretti/size(globalTestSet.targets,2);
testing.optimal.testSet.accuracy=correttiOpt/size(globalTestSet.targets,2);



dataSet.testSet.connMatrix=globalTestSet.connMatrix;
dataSet.testSet.maskMatrix=globalTestSet.maskMatrix;
dataSet.testSet.nodeLabels=globalTestSet.nodeLabels;
dataSet.testSet.nNodes=globalTestSet.nNodes;
dataSet.testSet.graphNum=globalTestSet.graphNum;
dataSet.testSet.targets=globalTestSet.targets;
if isfield(globalTestSet,'edgeLabels')
    dataSet.testSet.edgeLabels=globalTestSet.edgeLabels;
end
if isfield(globalTestSet,'nodesPerGraph')
    dataSet.testSet.nodesPerGraph=globalTestSet.nodesPerGraph;
end


% evaluate accuracy on graphs (instead of on nodes)
evaluateAccuracyOnGraphs('trainSet','current')
evaluateAccuracyOnGraphs('trainSet','optimal')
evaluateAccuracyOnGraphs('validationSet','current')
evaluateAccuracyOnGraphs('validationSet','optimal')
evaluateAccuracyOnGraphs('testSet','current')
evaluateAccuracyOnGraphs('testSet','optimal')

displayTestRes
