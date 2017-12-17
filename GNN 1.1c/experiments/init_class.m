function init_class(class)

makeGeneralDataset('trainSet',      strcat('@TesiDott/all_',class,'/train'),1);
makeGeneralDataset('validationSet', strcat('@TesiDott/all_',class,'/val'),  1);
makeGeneralDataset('testSet',       strcat('@TesiDott/all_',class,'/test'), 1);
removeCOGfromFeatures;
computeNormalizationFactors;
normalizeSet2;
Configure;
global dataSet;
eval(strcat('all_',class,'=dataSet; save all_',class,' all_',class));
