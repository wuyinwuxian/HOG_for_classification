%% 该函数是使用HOG来提取test_images下图片的特征的，代码编写参考matlab官方文档
% https://www.shangmayuan.com/a/5cb011b30b734328bac40194.html
%% 1.获取图片以及相关的分类
 
clear 
clc
currentPath = pwd;  % 获得当前的工作目录
 
imdsTrain = imageDatastore(fullfile(pwd,'train_images'),... 
    'IncludeSubfolders',true,... 
    'LabelSource','foldernames');   % 载入图片集合
 
%% 2 对训练集中的每张图像进行hog特征提取
% 预处理图像,主要是得到features特征大小，此大小与图像大小和Hog特征参数相关 
imageSize = [256,256];% 对所有图像进行此尺寸的缩放 
image1 = readimage(imdsTrain,1); 
scaleImage = imresize(image1,imageSize); 
features = extractHOGFeatures(scaleImage,'CellSize',[4,4]);
 
% 提示信息
disp('开始训练数据...');
% 对所有训练图像进行特征提取 
numImages = length(imdsTrain.Files); 
featuresTrain = zeros(numImages,size(features,2),'single'); % featuresTrain为单精度 
for i = 1:numImages 
    imageTrain = readimage(imdsTrain,i); 
    imageTrain = imresize(imageTrain,imageSize); 
    featuresTrain(i,:) = extractHOGFeatures(imageTrain,'CellSize',[4,4]); 
end 
 
% 所有训练图像标签 
trainLabels = imdsTrain.Labels; 
   
% 开始svm多分类训练，注意：fitcsvm用于二分类，fitcecoc用于多分类,1 VS 1方法 
% classifer = fitcecoc(featuresTrain,trainLabels); 
t = templateSVM('Standardize',true);
classifer = fitcecoc(featuresTrain,trainLabels,'Learners',t, 'ClassNames',{'daisy','roses','sunflowers'}); 

save classifer
% 把训练集带入的错误率
err = resubLoss(classifer)

% 10折交叉验证的错误率
CVMdl = crossval(classifer);
genError = kfoldLoss(CVMdl)
% 提示信息
disp('训练阶段结束！！！');