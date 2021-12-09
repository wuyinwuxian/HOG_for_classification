%% 该函数用来对图片进项分类 HOG + SVM
 
close all
clear
clc

%% 1.读入待分类的图片集合
currentPath = pwd;
imdsTest = imageDatastore(fullfile(pwd,'test_image'));
 
%% 2.分类，预测并显示预测效果图 
% 载入分类器
load classifer
 
numTest = length(imdsTest.Files); 
% correctCount:正确图片张数
% correctCount = 0;
 
for i = 1:numTest 
    testImage = readimage(imdsTest,i);  %  imdsTest.readimage(1)
    scaleTestImage = imresize(testImage,imageSize); 
    featureTest = extractHOGFeatures(scaleTestImage,'CellSize',[4,4]); 
    [predictIndex,score] = predict(classifer,featureTest); 
    figure;imshow(imresize(testImage,[256,256]));
     
    imgName = imdsTest.Files(i);
    tt = regexp(imgName,'\','split');
    cellLength =  cellfun('length',tt);
    tt2 = char(tt{1}(1,cellLength));
    % 统计正确率
%     if strfind(tt2,char(predictIndex))==1
%         correctCount = correctCount+1;
%     end
     
    title(['分类结果: ',tt2,'--',char(predictIndex)]); 
    fprintf('%s === %s \n',tt2,char(predictIndex));
end 
 
% 显示正确率
% fprintf('分类结束，正确率为：%.1f%%\n',correctCount * 100.0 / numTest);