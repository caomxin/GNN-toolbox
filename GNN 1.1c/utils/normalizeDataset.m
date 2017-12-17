function normalizeDataset
% This function normalize the dataSet, as values bigger than one are bad for learning
% The idea is to divide by 10 and cut to 1 values over 1 (i.e. values over 10 in the original dataset)
% Division by 10 is an idea to have less values cut to 1.


global dataSet
if isfield(dataSet,'validationSet')
	sets={'trainSet','validationSet','testSet'};
else
	sets={'trainSet','testSet'};
end

for s=1:size(sets,2)
	comp=find(max(dataSet.(sets{s}).nodeLabels,[],2)>1); %% problematic components
	dataSet.(sets{s}).nodeLabels(comp,:)=dataSet.(sets{s}).nodeLabels(comp,:)/10;
	[bigr bigc]=find(dataSet.(sets{s}).nodeLabels(comp,:)>1);
	bigr=comp(bigr);
	for i=1:size(bigr,1)
		dataSet.(sets{s}).nodeLabels(bigr(i),bigc(i))=1;
	end
	comp=find(max(dataSet.(sets{s}).edgeLabels,[],2)>1); %% problematic components
	dataSet.(sets{s}).edgeLabels(comp,:)=dataSet.(sets{s}).edgeLabels(comp,:)/10;
	[bigr bigc]=find(dataSet.(sets{s}).edgeLabels(comp,:)>1);
	bigr=comp(bigr);
	for i=1:size(bigr,1)
		dataSet.(sets{s}).edgeLabels(bigr(i),bigc(i))=1;
	end
end

