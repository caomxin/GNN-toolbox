function save_experiment(savefile)

test; global testing dataSet; 
saveDataSetResults(dataSet, 'trainSet', testing);
saveDataSetResults(dataSet, 'validationSet', testing);
saveDataSetResults(dataSet, 'testSet', testing);

if nargin==0
    savefile=input ('Input save file name\n >>> ', 's');
end
saveAll(strcat(fileparts(fileparts (dataSet.trainSet.srcdir)), '/', savefile))

