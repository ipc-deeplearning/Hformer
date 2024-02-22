%% 说明：该程序仿真动态颗粒群的同轴全息成像过程
% 输出颗粒群的全息图Holo和复振幅图CA；
% 颗粒是具有二维形状和一定厚度的颗粒；
% 静态颗粒粘在两个界面上，动态颗粒处于中间；
% 最终输出的全部信息mat包括颗粒位置x、y、z、折射率、相位图、振幅图、深度图；
% 上述单位定义：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数，um；
% 全息图为相对强度图，背景为1；图 = I/2*256
% 相位图为相对相位图，背景为0；
% 振幅图为强度衰减系数，背景为1；
% 深度图为各个颗粒的实际深度z，背景为0；图 = z/zmax*256

clear,clc
clear global

S = 1; % 总帧数

for s = 1:S

clear ss
ss = forwardAng;

%% 参数设置
N1 = round(unidrnd(20)/20*10+10); % 动态颗粒数
N2 = 2*unidrnd(10); % 静态颗粒数，偶数

zmin = 2000; % 样品最近距离，单位：um 
zmax = 8000; % 样品最远距离，单位：um
dz = 0.1;    % 深度切片精度，单位：um/pixel

lamda = 0.65;
FOV = 1024; % 视场大小（单方向），单位：pixel；
pixelpitch = 1.85; % 像素大小，单位：um；
n0 = 1; % 背景折射率
nr_min = 1.34; % 颗粒折射率范围，实部、虚部
nr_max = 1.35;
ni_min = 0.0003;
ni_max = 0.0005;

noise_var = 0.01; % 高斯噪声方差，仅Holo添加噪声

global maxdep_pp
    maxdep_pp = 10; %n 颗粒中心最大厚度，um

ss.opticaParams.zcam = zmin; % 重新定义到相机最近距离 单位：um 
ss.backGround.dz = 0.1; % 重新定义深度精度 单位：um/切片pixel 
ss.backGround.z = (zmax-zmin)/ss.backGround.dz; % 样品切片pixel总数 
ss.opticaParams.n0 = n0; % 重新定义背景折射率
ss.opticaParams.lamda = lamda; % 重新定义波长, um
ss.backGround.xy = [FOV, FOV]; % 重新定义视场大小, pixel
ss.opticaParams.nx = (2*ss.opticaParams.padlen+1)*ss.backGround.xy(1);
ss.opticaParams.ny = (2*ss.opticaParams.padlen+1)*ss.backGround.xy(2);
ss = gendf(ss,1);
ss.backGround.dxy = pixelpitch; % 重新定义像素大小, um

%%  创建文件夹
% 创建指定文件夹路径
folderPath = './eventData/time1';

HoloPath= [folderPath , '/Holo/'];
CAPath= [folderPath , '/CA/'];
HoloImgPath= [folderPath , '/Holo/Img/'];
PhaseImgPath= [folderPath , '/CA/Phase/'];
AmpImgPath= [folderPath , '/CA/Amp/'];
DepthImgPath= [folderPath , '/CA/Depth/'];

    mkdir(folderPath);

    mkdir(HoloImgPath);
    mkdir(PhaseImgPath);
    mkdir(AmpImgPath);
    mkdir(DepthImgPath);

%% 添加模型models到ss

imageFolder = './ShapeModels/pine';  % 取pine文件夹
filePattern = fullfile(imageFolder, '*.png');  % 适当的文件扩展名
% 获取文件列表
jpegFiles = dir(filePattern);
% 获取文件数量
numFiles = numel(jpegFiles);
% 生成随机排列的索引
randIndices = randperm(numFiles);
% 选择前20个随机排列的文件
numToSelect = min(numFiles, 20); % 防止选择超过文件数量的文件
selectedIndices = randIndices(1:numToSelect);

for i = selectedIndices % 取pine文件夹中随机取图
    baseFileName = jpegFiles(i).name;
    fullFileName = fullfile(imageFolder, baseFileName);
    I = imread(fullFileName); % 调整模型深度
    I = modeldepth_adjust(I); % 调整模型深度
    ss = addmodels(ss,I); % 添加模型
end

imageFolder = './ShapeModels/peach';  % 取peach文件夹
filePattern = fullfile(imageFolder, '*.png');  % 适当的文件扩展名
% 获取文件列表
jpegFiles = dir(filePattern);
% 获取文件数量
numFiles = numel(jpegFiles);
% 生成随机排列的索引
randIndices = randperm(numFiles);
% 选择前20个随机排列的文件
numToSelect = min(numFiles, 20); % 防止选择超过文件数量的文件
selectedIndices = randIndices(1:numToSelect);

for i = selectedIndices % 取peach文件夹中随机取图
    baseFileName = jpegFiles(i).name;
    fullFileName = fullfile(imageFolder, baseFileName);
    I = imread(fullFileName);
    I = modeldepth_adjust(I); % 调整模型深度
    ss = addmodels(ss,I); % 添加模型
end

% 设置图像文件夹和文件扩展名
imageFolder = './ShapeModels/corn';  % 取corn文件夹
filePattern = fullfile(imageFolder, '*.png');  % 适当的文件扩展名
% 获取文件列表
jpegFiles = dir(filePattern);
% 获取文件数量
numFiles = numel(jpegFiles);
% 生成随机排列的索引
randIndices = randperm(numFiles);
% 选择前20个随机排列的文件
numToSelect = min(numFiles, 20); % 防止选择超过文件数量的文件
selectedIndices = randIndices(1:numToSelect);

for i = selectedIndices
    baseFileName = jpegFiles(i).name;
    fullFileName = fullfile(imageFolder, baseFileName);
    I = imread(fullFileName);
    I = modeldepth_adjust(I); % 调整模型深度
    ss = addmodels(ss,I); % 添加模型
end

Holo = cell(0); % 全息图
CA = cell(0); % 颗粒所有信息

%% 添加颗粒samples到ss

 % 从所有模型中随机选取，先添加N1个动态颗粒，再添加N2个静态颗粒
ss = addsamples(ss,N1,'type','mask','R',[1,length(ss.models)],'n',[nr_min,ni_min;nr_max,ni_max]);
ss = addsamples(ss,N2,'type','mask','R',[1,length(ss.models)],'n',[nr_min,ni_min;nr_max,ni_max]);

% 重新指定静态S颗粒位置xy，单位pixel，后面N/2个颗粒静态
% ss.samples.x(floor(n/2)+1:end,:) = [50,200,300,400,500,150,250,350,450,550]; 

% 指定静态颗粒两种极端深度z
ss.samples.z(N1+1:floor(N1+N2/2),:) = ss.backGround.z-(zmax-ss.opticaParams.zcam)/ss.backGround.dz+1; % 不等于0
ss.samples.z(floor(N1+N2/2)+1:end,:) = ss.backGround.z-(zmin-ss.opticaParams.zcam)/ss.backGround.dz; % 切片总数

%% 输出全息图及各种标签

    ss = forwardV(ss); % 前向传播
    [ss,cam1] = forwardCam(ss);
    Holo = abs(gather(cam1)); % 静态颗粒前向传播
    PhaseMap = zeros(FOV);
    AmpMap = zeros(FOV);
    DepthMap = zeros(FOV);
    for j = 1:length(ss.samples.x)
        % CA 各列分别为 x,y,z,n,PhaseMap, AmpMap. 
        % 单位：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数
        CA{j,1} = ss.samples.x(j);
        CA{j,2} = ss.samples.y(j);
        CA{j,3} = (ss.backGround.z-ss.samples.z(j))*ss.backGround.dz+ss.opticaParams.zcam; % 距离相机实际位置，um
        CA{j,4} = ss.samples.n(j);
        index = ss.models{round(ss.samples.R(j))};
        [CA{j,5},CA{j,6},CA{j,7}] = PhaseSlice(FOV,...
            ss.samples.x(j),...
            ss.samples.y(j),...
            CA{j,3},... % 距离相机实际位置，um
            index,...
            ss.samples.n(j),...
            n0,lamda);
        PhaseMap = PhaseMap + CA{j,5};
        AmpMap =  AmpMap + CA{j,6};
        DepthMap =  DepthMap + CA{j,7};
    end

    filename1 = [HoloPath,num2str(s,'%03d'),'.mat'];
    filename2 = [HoloImgPath,num2str(s,'%03d'),'.png'];
    filename3 = [CAPath,num2str(s,'%03d'),'.mat'];
    filename4 = [PhaseImgPath,num2str(s,'%03d'),'.png'];
    filename5 = [AmpImgPath,num2str(s,'%03d'),'.png'];
    filename6 = [DepthImgPath,num2str(s,'%03d'),'.png'];

    save(filename1, "Holo"); %  颗粒全息图，强度图
    imwrite(imnoise(uint8((Holo/2 * 256)),'gaussian',0,noise_var), filename2); 
    %  强度到灰度：/2*256，单位：nan，增加高斯白噪声noise_var
    save(filename3, "CA","PhaseMap","AmpMap","DepthMap"); %  颗粒所有信息
    imwrite(uint8((PhaseMap)/pi*256), filename4); %  多层叠加相位图，基于pi归一化，单位nan
    imwrite(uint8((AmpMap)/1*256), filename5); %  多层叠加振幅图，光强衰减系数，单位nan
    imwrite(uint8((DepthMap)/zmax*256), filename6); %  多层叠加深度图，越亮越远，单位um
end

