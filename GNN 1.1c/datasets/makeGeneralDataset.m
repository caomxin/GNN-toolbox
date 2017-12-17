function makeGeneralDataset(set,srcdir,loadTargets)
t=cputime;
if nargin~=3
    disp('Usage: makeGeneraldataset(''set'',''sourcedir'',''loadTargets'')')
    return
end
if ~strcmp(set,'trainSet') && ~strcmp(set,'validationSet') && ~strcmp(set,'testSet')
    err(0,'first argument must be in {trainSet,validationSet, testSet}')
    return
end
if loadTargets~=0 && loadTargets ~=1
    err(0,'third argument must be 1 if targets must be read from files, 0 if centralRegionId must be used')
    return
end
global dataSet

origDir=pwd;
try
    cd(srcdir)
catch
    err(0,[srcdir ' is not a valid source directory']);
    return
end
if srcdir(end)==filesep     %if srcdir has / or \ in the last position, remove it
    srcdir=srcdir(1:end-1);
end

dataSet.(set).connMatrix=sparse([],[],[]);
dataSet.(set).maskMatrix=sparse([],[],[]);
dataSet.(set).nodeLabels=[];
dataSet.(set).edgeLabels=[];
dataSet.(set).targets=[];
dataSet.(set).nodesPerGraph=[];
dataSet.(set).srcdir=srcdir;

dataSet.config.type='classification';
dataSet.config.rejectUpperThreshold=0;
dataSet.config.rejectLowerThreshold=0;
dataSet.config.nodeLabelsDim=23;
dataSet.config.edgeLabelsDim=5;


posnegfs=1;     % if 1 load files from <srcdir>/positive <srcdir>/negative
% else from <srcdir>
try
    cd positive
    cd ../negative
    cd ..
catch
    posnegfs=0;
end




if posnegfs
    a=dir('positive/*matrix*');
    numpos=size(a,1);
    for i=1:numpos
        posfiles{i}=a(i).name(1:6);    % extract all the names
    end
    a=dir('negative/*matrix*');
    numpos=size(a,1);
    for i=1:numpos
        negfiles{i}=a(i).name(1:6);    % extract all the names
    end
    dataSet.(set).graphNum=size(posfiles,2)+size(negfiles,2);
    %positive files
    cd positive
    for i=1:size(posfiles,2)
        tmp=load([char(posfiles(i)) '_matrix.txt']);
        sz=size(tmp,1);
        dataSet.(set).nodesPerGraph(end+1)=sz;
        dataSet.(set).connMatrix(end+1:end+sz,end+1:end+sz)=tmp;
        if loadTargets
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=eye(sz);
        else
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=zeros(sz);
        end
        tmp=load([char(posfiles(i)) '_nodeLabels.txt']);
        dataSet.(set).nodeLabels(:,end+1:end+sz)=tmp';
        if ~loadTargets
            tmp=load([char(posfiles(i)) '_centralRegionId.txt']);
            tgindex = size(dataSet.(set).targets,2)+tmp+1;
            if tmp >= sz
                error(['tmp=' num2str(tmp) ' sz=' num2str(sz) ' ' srcdir '/positive/' char(posfiles(i))]);
            end
            dataSet.(set).maskMatrix(tgindex,tgindex)=1;
            dataSet.(set).targets(1,end+1:end+sz)=zeros(1,sz);
            dataSet.(set).targets(tgindex)=1;
        else
            if exist([char(posfiles(i)) '_targets.txt'])
                dataSet.(set).targets(1,end+1:end+sz)=load([char(posfiles(i)) '_targets.txt']);
            else
                dataSet.(set).targets(1,end+1:end+sz)=-ones(1,sz);
            end
        end           
        tmp=load([char(posfiles(i)) '_edgeLabels.txt']);
        dataSet.(set).edgeLabels(:,end+1:end+size(tmp,1))=tmp(:,3:end)';
    end
    %negative files
    cd ../negative
    for i=1:size(negfiles,2)
        tmp=load([char(negfiles(i)) '_matrix.txt']);
        sz=size(tmp,1);
        dataSet.(set).nodesPerGraph(end+1)=sz;
        dataSet.(set).connMatrix(end+1:end+sz,end+1:end+sz)=tmp;
        if loadTargets
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=eye(sz);
        else
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=zeros(sz);
        end
        tmp=load([char(negfiles(i)) '_nodeLabels.txt']);
        dataSet.(set).nodeLabels(:,end+1:end+sz)=tmp';
        if ~loadTargets
            tmp=load([char(negfiles(i)) '_centralRegionId.txt']);
            tgindex = size(dataSet.(set).targets,2)+tmp+1;
            if tmp >= sz
                error(['tmp=' num2str(tmp) ' sz=' num2str(sz) ' ' srcdir '/negative/' char(negfiles(i))]);
            end
            dataSet.(set).maskMatrix(tgindex,tgindex)=1;
            dataSet.(set).targets(1,end+1:end+sz)=zeros(1,sz);
            dataSet.(set).targets(tgindex)=-1;
        else
            if exist([char(negfiles(i)) '_targets.txt'])
                dataSet.(set).targets(1,end+1:end+sz)=load([char(negfiles(i)) '_targets.txt']);
            else
                dataSet.(set).targets(1,end+1:end+sz)=-ones(1,sz);
            end
        end    
        tmp=load([char(negfiles(i)) '_edgeLabels.txt']);
        dataSet.(set).edgeLabels(:,end+1:end+size(tmp,1))=tmp(:,3:end)';
    end
    dataSet.(set).nNodes=size(dataSet.(set).connMatrix,1);
    cd ..
    
    %test for correctness (only for Linux users)
    if ~ispc'
        [status,tNodes]=system('cat {positive,negative}/*_matrix* | wc -l');
        tNodes=str2Int(tNodes);
        tNodes=tNodes+dataSet.(set).graphNum;
        if dataSet.(set).nNodes~=tNodes
            err(0,['size(nodeLabels,2)=' num2str(dataSet.(set).nNodes) ' but tNodes=' num2str(tNodes)]);
            dataSet.(set)=[];
            return
        end
        cmd=['python countTotEdges.py positive/'];    % final slash is crucial for Python script to work properly!
        [status,tEdges]=system(cmd);
        tEdges=str2Int(tEdges);
          cmd=['python countTotEdges.py negative/'];    % final slash is crucial for Python script to work properly!
        [status,tEdges2]=system(cmd);
        tEdges=tEdges+str2Int(tEdges2);
        if size(dataSet.(set).edgeLabels,2)~=tEdges
            err(0,['size(edgeLabels,2)=' num2str(size(dataSet.(set).edgeLabels,2)) ' but tEdges=' num2str(tEdges)]);
            dataSet.(set)=[];
            return
        end
    end
else % posnegfs=0
    a=dir('*_matrix.txt');
    numpos=size(a,1);
    for i=1:numpos
        files{i}=a(i).name(1:end-size('_matrix.txt',2));    % extract all the names
    end
    dataSet.(set).files=files;
    dataSet.(set).graphNum=size(files,2);
    for i=1:size(files,2)
        tmp=load([char(files(i)) '_matrix.txt']);
        sz=size(tmp,1);
        dataSet.(set).nodesPerGraph(end+1)=sz;
        dataSet.(set).connMatrix(end+1:end+sz,end+1:end+sz)=tmp;
        if loadTargets
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=eye(sz);
        else
            dataSet.(set).maskMatrix(end+1:end+sz,end+1:end+sz)=zeros(sz);
        end
        tmp=load([char(files(i)) '_nodeLabels.txt']);
        dataSet.(set).nodeLabels(:,end+1:end+sz)=tmp';
        if ~loadTargets
            tmp=load([char(files(i)) '_centralRegionId.txt']);
            tgindex = size(dataSet.(set).targets,2)+tmp+1;
            if tmp >= sz
                error(['tmp=' num2str(tmp) ' sz=' num2str(sz) ' ' srcdir '/positive/' char(files(i))]);
            end
            dataSet.(set).maskMatrix(tgindex,tgindex)=1;
            dataSet.(set).targets(1,end+1:end+sz)=zeros(1,sz);
            dataSet.(set).targets(tgindex)=1;
        else
            s=pwd;
            if exist([s '/' char(files(i)) '_targets.txt'])
                %disp('exist');
                dataSet.(set).targets(1,end+1:end+sz)=load([s '/' char(files(i)) '_targets.txt']);
            else
                dataSet.(set).targets(1,end+1:end+sz)=-ones(1,sz);
            end
        end
        tmp=load([char(files(i)) '_edgeLabels.txt']);
        dataSet.(set).edgeLabels(:,end+1:end+size(tmp,1))=tmp(:,3:end)';
    end
    dataSet.(set).nNodes=size(dataSet.(set).connMatrix,1);

    %test for correctness (only for Linux users
    if ~ispc'
        [status,tNodes]=system('cat *_matrix* | wc -l');
        tNodes=str2Int(tNodes);
        tNodes=tNodes+dataSet.(set).graphNum;
        if dataSet.(set).nNodes~=tNodes
            err(0,['size(nodeLabels,2)=' num2str(dataSet.(set).nNodes) ' but tNodes=' num2str(tNodes)]);
            dataSet.(set)=[];
            return
        end
        cmd=['python countTotEdges.py ./'];    % final slash is crucial for Python script to work properly!
        [status,tEdges]=system(cmd);
        if ~status
            tEdges=str2Int(tEdges);
            if size(dataSet.(set).edgeLabels,2)~=tEdges
                err(0,['size(edgeLabels,2)=' num2str(size(dataSet.(set).edgeLabels,2)) ' but tEdges=' num2str(tEdges)]);
                dataSet.(set)=[];
                return
            end
        else
            warn(0,['I can''t find python script countTotEdges.py in ' pwd])
        end
    end
end


cd(origDir)
t=cputime-t;
message1(['Added to ' (set) ' ' num2str(dataSet.(set).graphNum) ' graphs (' num2str(dataSet.(set).nNodes) ' nodes, ' num2str(size(dataSet.(set).edgeLabels,2)) ' edges) in ' num2str(t) 's.']);

