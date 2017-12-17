function makeMutagenicDataset(file)
% Create a dataset from the Mutagenic Dataset
% USAGE: makeMutagenicDataset
%
% A bug was removed from this procedure in version 1.1b. In the previous versions,  
% the splitting of the Mutagenesis dataset for the ten fold cross
% validation procedure was not correct.
% We thank Derek Monner to have shown us the problem.


the Procedure  procedure 
addpath(genpath(pwd));

global dataSet tmp gr multidata testIndexes trainIndexes validationIndexes
tmp=[];
gr=[];
if ~isempty(dataSet)
    dataSet=[];
end


tmp.graphstarts=1;                     %record starting node of each graph


tmp.numGlobalFeatures=4;
tmp.numDifferentAtoms=9;
tmp.numAtomFeatures=2;
tmp.nodeLabelsDim=tmp.numGlobalFeatures+tmp.numDifferentAtoms+tmp.numAtomFeatures;
tmp.nodeLabels=[];
tmp.targets=[];
tmp.connMatrix=sparse([],[],[]);

%delimiter = [';' ' ' 9 13];         %ASCII(9) = TABS, ASCII(13) = carriage return


file='muta.pl';
% parse file
%try
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end

    numline=1;
    line = fgetl(fid);
    while feof(fid) == 0
        numline
    %for ttt=1:2
        if isempty(line)
            err(numline, 'line is empty. Please remove it');
        elseif line([1 2])=='%!'
            %disp(['line ' num2str(numline) ': I''ve found a new graph']);
            numatom=0;
            gr.nodeLabels=[];
            gr.currentGlobalFeatures=zeros(tmp.numGlobalFeatures,1);

            %gestione target
            line = fgetl(fid);
            if isempty(strfind(line,'yes'))
                gr.targets=-1;
            else
                gr.targets=1;
            end

            % global features
            start=[]; stop=[];
            for i=1:tmp.numGlobalFeatures
                line = fgetl(fid);
                start=strfind(line,',')+1;
                stop=strfind(line,')')-1;
                value=line(start:stop);
                if strcmp(value,'false')
                    gr.currentGlobalFeatures(i)=0.0;
                elseif strcmp(value,'true')
                    gr.currentGlobalFeatures(i)=1.0;
                else
                    gr.currentGlobalFeatures(i)=str2num(value);
                end
            end
            %tmp.currentGlobalFeatures
            numline=numline+tmp.numGlobalFeatures;

            line = fgetl(fid);
            numline=numline+1;
            while line(1)=='m'
                numline=numline+1;
                numatom=numatom+1;
                %skip lines that only give atom names
                line = fgetl(fid);
            end
            gr.nodeLabels(1:tmp.nodeLabelsDim,end+1:end+numatom)=zeros(tmp.nodeLabelsDim,numatom);
            gr.targets(2:numatom)=zeros(1,numatom-1);
            gr.connMatrix=sparse(numatom,numatom);

            gr.nodeLabels((tmp.numDifferentAtoms+tmp.numAtomFeatures+1):tmp.nodeLabelsDim,:)=repmat(gr.currentGlobalFeatures,1,numatom);
            start=[]; stop=[];
            for f=0:numatom-1
                %line
                start=strfind(line,'_');
                stop=strfind(line,',');
                stop(end+1)=strfind(line,')');
                ind=str2num(line(start(2)+1:stop(1)-1));
                value=zeros(1,tmp.numDifferentAtoms);
                str=line(stop(1)+1:stop(2)-1);
                switch str
                    case 'br'
                        value=[0 0 0 0 0 0 0 0 1];
                    case 'c'
                        value=[0 0 0 0 0 0 0 1 0];
                    case 'cl'
                        value=[0 0 0 0 0 0 1 0 0];
                    case 'f'
                        value=[0 0 0 0 0 1 0 0 0];
                    case 'h'
                        value=[0 0 0 0 1 0 0 0 0];
                    case 'i'
                        value=[0 0 0 1 0 0 0 0 0];
                    case 'n'
                        value=[0 0 1 0 0 0 0 0 0];
                    case 'o'
                        value=[0 1 0 0 0 0 0 0 0];
                    case 's'
                        value=[1 0 0 0 0 0 0 0 0];
                    otherwise
                        str
                        warn(numline,'unrecognized atoms');
                end
                gr.nodeLabels(1:tmp.numDifferentAtoms,ind)=value;
                for j=2:3
                    str=line(stop(j)+1:stop(j+1)-1);
                    value=str2num(str);
                    if isempty(value)
                        err(0,'value is empty!');
                    end
                    gr.nodeLabels(j+tmp.numDifferentAtoms-1,ind)=value;
                end
                line=fgetl(fid);
                numline=numline+1;
            end

            %bond line
            while line(1)=='b'
                start=[];
                stop=[];
                start=strfind(line,'_');
                stop=strfind(line,',');

                %line
                %line(start(1)+1:stop(1)-1)
                %line(start(2)+1:stop(2)-1)

                gr.connMatrix(str2num(line(start(1)+1:stop(1)-1)),str2num(line(start(2)+1:stop(2)-1)))=1;
                gr.connMatrix(str2num(line(start(2)+1:stop(2)-1)),str2num(line(start(1)+1:stop(1)-1)))=1;

                numline=numline+1;
                line = fgetl(fid);
            end
            %line = strtok(line, delimiter);
            %k = strfind(line,'=');
            %if size(k,2)==0
            %    err(numline,'No ''='' detected');
            %elseif (size(k,2)>1)
            %    err(numline,'More than one "=" detected');
            %end
            %name = line(1:k-1);
            %value = line(k+1:end);
            %check_valueok(name,value,num2str(numline));
        end


        %create global structure tmp
        tmp.graphstarts(end+1)=tmp.graphstarts(end)+numatom;
        tmp.nodeLabels(1:tmp.nodeLabelsDim,end+1:end+numatom)=gr.nodeLabels;
        tmp.targets(end+1:end+numatom)=gr.targets;
        tmp.connMatrix(end+1:end+numatom,end+1:end+numatom)=gr.connMatrix;


    end
    fclose(fid);
    %check_thereisall;
% catch
%     thiserr=lasterror;
%     thiserr.message
%     err(0,'error');
%     fclose all;
%     return;
% end


tmp.maskMatrix=sparse(zeros(size(tmp.connMatrix,1)));
for y=1:size(tmp.graphstarts,2)
    tmp.maskMatrix(tmp.graphstarts(y),tmp.graphstarts(y))=1;
end







%randgraphs=1:size(tmp.graphstarts,2)-1;
randgraphs=randperm(size(tmp.graphstarts,2)-1);



global trainIndexes validationIndexes testIndexes

initialTestNumGraph=floor((size(tmp.graphstarts,2)-1)/10);
skewnumber=size(tmp.graphstarts,2)-1-initialTestNumGraph*10; % numero di fold che si beccano un grafo in piu'



% ten cross validation: build all ten datasets
for dd=1:10

    trainIndexes=[];
    validationIndexes=[];
    testIndexes=[];
    
    multidata(dd).config.type='classification';

    multidata(dd).config.nodeLabelsDim=tmp.nodeLabelsDim;
    multidata(dd).config.rejectLowerThreshold=0;
    multidata(dd).config.rejectUpperThreshold=0;

    multidata(dd).testSet.graphNum=initialTestNumGraph;
    if (dd<=skewnumber) multidata(dd).testSet.graphNum=multidata(dd).testSet.graphNum+1; end % i primi skewnumber folder si beccano un grafo in piu'
    
    multidata(dd).validationSet.graphNum=floor((size(tmp.graphstarts,2)-1)*0.2);
    multidata(dd).trainSet.graphNum=size(tmp.graphstarts,2)-1-multidata(dd).validationSet.graphNum-multidata(dd).testSet.graphNum;

    
    
    gra=((dd-1)*multidata(dd).testSet.graphNum+1):dd*multidata(dd).testSet.graphNum;
    for g=gra(1):gra(end)
        ind=tmp.graphstarts(randgraphs(g)):tmp.graphstarts(randgraphs(g)+1)-1;
        testIndexes(end+1:end+size(ind,2))=ind;
    end
    
    

    % Re-order these to ensure the validation set is not the same across folds
    remaining=[gra(end)+1:size(randgraphs,2) 1:gra(1)-1];
    %remaining=[1:gra(1)-1 gra(end)+1:size(randgraphs,2)];

    % Instead of ranging between two indices in "remaining", select the indices in "remaining" with a certain index range.
    % One must do this because "remaining" will contain gaps where the test set was removed.
    for rem=remaining(1:multidata(dd).trainSet.graphNum)
    %for rem=remaining(1):remaining(multidata(dd).trainSet.graphNum)
        %rem
        ind=tmp.graphstarts(randgraphs(rem)):tmp.graphstarts(randgraphs(rem)+1)-1;
        trainIndexes(end+1:end+size(ind,2))=ind;
    end
    %disp('**************************************************************************')
    % Instead of ranging between two indices in "remaining", select the indices in "remaining" with a certain index range.
    % One must do this because "remaining" will contain gaps where the test set was removed.
    for rem=remaining(multidata(dd).trainSet.graphNum+1:end)
    %for rem=remaining(multidata(dd).trainSet.graphNum+1):remaining(end)
        %rem
        ind=tmp.graphstarts(randgraphs(rem)):tmp.graphstarts(randgraphs(rem)+1)-1;
        validationIndexes(end+1:end+size(ind,2))=ind;
    end

    % Shows the total size of each of the training, validation, and test sets. They should be at a (roughly) 7:2:1 size ratio.
    disp(['train/valid/test sizes: ' int2str(size(trainIndexes,2)) ' ' int2str(size(validationIndexes,2)) ' ' int2str(size(testIndexes,2))])
    % Show the sum of the above sizes. Since the three sets do not overlap, this should be equal to the size of the "unique elements only" set below.
    disp(['total size: ' int2str(size([trainIndexes validationIndexes testIndexes], 2))])
    % Show the total number of unique elements in all three sets. If this does not equal the "total size" above, then the sets overlap.
    disp(['unique size: ' int2str(size(unique(sort([trainIndexes validationIndexes testIndexes])),2))])


    multidata(dd).trainSet.nNodes=size(trainIndexes,2);
    multidata(dd).validationSet.nNodes=size(validationIndexes,2);
    multidata(dd).testSet.nNodes=size(testIndexes,2);


    %In the following the graphs are subdivided into validation, test and
    %training set

    multidata(dd).trainSet.maskMatrix=tmp.maskMatrix(trainIndexes,trainIndexes);
    multidata(dd).trainSet.connMatrix=tmp.connMatrix(trainIndexes,trainIndexes);
    multidata(dd).trainSet.nodeLabels=tmp.nodeLabels(:,trainIndexes);
    multidata(dd).trainSet.targets=tmp.targets(:,trainIndexes);

    multidata(dd).testSet.maskMatrix=tmp.maskMatrix(testIndexes,testIndexes);
    multidata(dd).testSet.connMatrix=tmp.connMatrix(testIndexes,testIndexes);
    multidata(dd).testSet.nodeLabels=tmp.nodeLabels(:,testIndexes);
    multidata(dd).testSet.targets=tmp.targets(:,testIndexes);

    multidata(dd).validationSet.maskMatrix=tmp.maskMatrix(validationIndexes,validationIndexes);
    multidata(dd).validationSet.connMatrix=tmp.connMatrix(validationIndexes,validationIndexes);
    multidata(dd).validationSet.nodeLabels=tmp.nodeLabels(:,validationIndexes);
    multidata(dd).validationSet.targets=tmp.targets(:,validationIndexes);


    

end


% for gra=1:dataSet.trainSet.graphNum
%     ind=tmp.graphstarts(randgraphs(gra)):tmp.graphstarts(randgraphs(gra)+1)-1;
%     trainIndexes(end+1:end+size(ind,2))=ind;
% end
% for gra=dataSet.trainSet.graphNum+1:dataSet.trainSet.graphNum+dataSet.validationSet.graphNum
%     ind=tmp.graphstarts(randgraphs(gra)):tmp.graphstarts(randgraphs(gra)+1)-1;
%     validationIndexes(end+1:end+size(ind,2))=ind;
% end
% for gra=dataSet.trainSet.graphNum+dataSet.validationSet.graphNum+1:dataSet.trainSet.graphNum+dataSet.validationSet.graphNum+dataSet.testSet.graphNum
%     ind=tmp.graphstarts(randgraphs(gra)):tmp.graphstarts(randgraphs(gra)+1)-1;
%     testIndexes(end+1:end+size(ind,2))=ind;
% end


%tmp.graphstarts=tmp.graphstarts(1:end-1);   %discard last element



% dataSet.trainSet.nNodes=size(trainIndexes,2);
% dataSet.validationSet.nNodes=size(validationIndexes,2);
% dataSet.testSet.nNodes=size(testIndexes,2);
% 
% 
% %In the following the graphs are subdivided into validation, test and
% %training set
% 
% dataSet.trainSet.maskMatrix=tmp.maskMatrix(trainIndexes,trainIndexes);
% dataSet.trainSet.connMatrix=tmp.connMatrix(trainIndexes,trainIndexes);
% dataSet.trainSet.nodeLabels=tmp.nodeLabels(:,trainIndexes);
% dataSet.trainSet.targets=tmp.targets(:,trainIndexes);
% 
% dataSet.testSet.maskMatrix=tmp.maskMatrix(testIndexes,testIndexes);
% dataSet.testSet.connMatrix=tmp.connMatrix(testIndexes,testIndexes);
% dataSet.testSet.nodeLabels=tmp.nodeLabels(:,testIndexes);
% dataSet.testSet.targets=tmp.targets(:,testIndexes);
% 
% dataSet.validationSet.maskMatrix=tmp.maskMatrix(validationIndexes,validationIndexes);
% dataSet.validationSet.connMatrix=tmp.connMatrix(validationIndexes,validationIndexes);
% dataSet.validationSet.nodeLabels=tmp.nodeLabels(:,validationIndexes);
% dataSet.validationSet.targets=tmp.targets(:,validationIndexes);



