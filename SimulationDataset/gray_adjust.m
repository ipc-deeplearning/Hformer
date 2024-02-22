function U = gray_adjust(I,fun)
%GRAY_ADJUST - 校正图像灰度
%   此函数返回灰度校正后的图片U，它是图像I通过除以二次函数fun(I)所得到的
%   灰度校正图像
%
%   U = GRAY_ADJUST(I) 校正图像I的灰度
%   U = GRAY_ADJUST(I,fun) 使用函数fun校正图像I的灰度
%
%   输入参数
%       I - 输入图像
%           double
%       fun - 校正函数
%           function_handle  | @(x) -3.2852e-04*x.^2+0.0022*x+0.0044(默认值)
%
%   输出参数
%       U - 矫正后图像
%           double
%
%   另请参阅
%
%MATLAB2022b - 2023.4.25 - by SZU-IPC
    arguments
        I double % 输入图像
        fun = @(x) -3.2852e-04*x.^2+0.0022*x+0.0044 % 校正函数
    end
    %% 输出校正的强度图
    output=uint8((I)./fun(I));
    U = im2double(output+25.763);
end

