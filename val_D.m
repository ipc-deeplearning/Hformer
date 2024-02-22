
clear;
clc;

imageD = cell(1, 50);
start=1;
path1 = 'D:\res\ynet\yfS\';
for i = start:50
         imageD{i} = imread([path1,num2str(i+1,'%03d'),'D.png']);  
    
end


%读取真值


% for i = 1:50
%     imageD{i} = imread(['D:\czx\ynetRes\D\', num2str(i-1), '.png']); % 假设图像文件名为image1.jpg, image2.jpg, ..., image50.jpg
%     
% end

for i = start:50
         imagePreD{i-start+1} = imread(['D:\res\ynet\Time7\CAS\Depth\001.png']);
        imagePreD{i-start+1} = imagePreD{i-start+1}(31:31+255,31:31+255,:);

    
end


% 创建一个矩阵来存储SSIM值
ssimValues = zeros(50);
psnrValues = zeros(50);
mseValues = zeros(50);

% 计算SSIM值
for i = 1:50
%          psnrValues(i) = psnr(imageD{i}, imagePreD{i});
         ssimValues(i) = ssim(imageD{i}(:,:,1), imagePreD{i}(:,:,1));
          psnrValues(i) = psnr(imageD{i}(:,:,1), imagePreD{i}(:,:,1));

         mseValues(i)= mean(mse(imageD{i}(:,:,1)-imagePreD{i}(:,:,1)));


end

mse = mean(mseValues(:,1));
ssimRes = mean(ssimValues(:,1));
psnrRes = mean(psnrValues(:,1));

