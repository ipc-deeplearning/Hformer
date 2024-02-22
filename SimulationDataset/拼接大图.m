% 设置输入文件夹路径
folder1 = 'D:\newF\TimingData9\HoloN\Img\';
folder2 = 'D:\res\1024noNoise\';
folder3 = 'D:\res\reallyDepth\';
folder4 = 'D:\res\fakerD\';

   output_path = './folder';
% 读取每个文件夹下的图像
num_images = 400;

% 创建一个 2048x2048 的大图
%边缘10
by=200;
by2=350;
big_image = ones(2048+by, 2048+by2, 'uint8')*255;

for i = 1:num_images
    % 图像名字从 "001" 到 "300"
    image_name = sprintf('%03d', i);
    
    images1 = imread(fullfile(folder1, [image_name '.png']));
        image_name1 = sprintf('%03d', i-1);
    images2 = imread(fullfile(folder2, [image_name1 '.png']));
    images3 = imread(fullfile(folder3, [image_name '.png']));
    images4 = imread(fullfile(folder4, [image_name '.png']));

        big_image(1:1024, 1:1024, :) = images1;
    big_image(1:1024, 1025+by2:2048+by2, :) = images2(:,:,1);
    big_image(1025+by:2048+by,1:1024, :) = images3(:,:,1);
    big_image(1025+by:2048+by, 1025+by2:2048+by2, :) = images4(:,:,1);
    % 保存拼接后的图像

    output_filename = sprintf('%03d', i);
    imwrite(big_image, fullfile(output_path, [output_filename '.png']));
end




%%




