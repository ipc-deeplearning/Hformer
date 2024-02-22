% 指定图像所在文件夹
inputFolder = 'D:\res\cutPicture\time5';  % 替换为你的文件夹路径
outputFolder = 'D:\res\cutPicture\timeRes\time5Predict';  % 替换为你的输出文件夹路径

% 获取文件夹下所有JPEG图片的文件列表
imageFiles = dir(fullfile(inputFolder, '*.png'));

% 遍历每张图片并进行截取
for i = 1:length(imageFiles)
    % 读取原始图像
    originalImage = imread(fullfile(inputFolder, imageFiles(i).name));

    % 截取图像
    x = 512;  % 截取区域的左上角 x 坐标
    y = 512;  % 截取区域的左上角 y 坐标
    width = 512;  % 截取区域的宽度
    height = 512;  % 截取区域的高度

    croppedImage = imcrop(originalImage, [x, y, width - 1, height - 1]);

    % 保存截取后的图像到输出文件夹
    [~, baseFileName, ~] = fileparts(imageFiles(i).name);
    outputFileName = fullfile(outputFolder, [baseFileName '.png']);
    imwrite(croppedImage, outputFileName);
end
