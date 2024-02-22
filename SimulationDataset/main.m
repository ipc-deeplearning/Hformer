%% 说明：该程序仿真动态颗粒群的同轴全息成像过程
% 输出颗粒群的全息图Holo和复振幅图CA；
% 颗粒是具有二维形状和一定厚度的颗粒；
% 全息图和复振幅图均包含动态D、静态S、全局N三种情况；
% 静态颗粒粘在两个界面上，动态颗粒处于中间；
% 最终输出的全部信息mat包括颗粒位置x、y、z、折射率、相位图、振幅图、深度图；
% 上述单位定义：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数；
% 全息图为相对强度图，背景为1；
% 相位图为相对相位图，背景为0；
% 振幅图为强度衰减系数，背景为1；

clear,clc
clear global
ss = forwardAng;

%% 参数设置
T = 300; % 总帧数
N1 = 10; % 动态颗粒数
N2 = 12; % 静态颗粒数，取偶数
vmax = 6; % 动态颗粒的最大速度，pixel/帧
vmin = 3; % 动态颗粒的最小速度，pixel/帧
v = floor(unidrnd(vmax,N1,2)/vmax*(vmax-vmin)+vmin); % 动态颗粒的速度，pixel/帧

zmin = 2000; % 样品最近距离，单位：um 
zmax = 8000; % 样品最远距离，单位：um 

lamda = 0.68;
FOV = 1024; % 视场大小（单方向），单位：pixel；
pixelpitch = 1.85; % 像素大小，单位：um；
n0 = 1; % 背景折射率
nr_min = 1.34; % 颗粒折射率范围，实部、虚部
nr_max = 1.35;
ni_min = 0.0003;
ni_max = 0.0005;

global maxdep_pp
    maxdep_pp = 10; %n 颗粒中心最大厚度，um

ss.opticaParams.zcam = zmin; % 重新定义到相机最近距离 单位：um 
ss.backGround.dz = 0.1; % 重新定义深度精度 单位：um/切片pixel 
ss.backGround.z = (zmax-zmin)/ss.backGround.dz; % 样品切片pixel总数 
ss.opticaParams.n0 = n0; % 重新定义背景折射率
ss.opticaParams.lamda = lamda; % 重新定义波长, um
ss.backGround.xy = [FOV, FOV]; % 重新定义视场大小, pixel
ss.backGround.dxy = pixelpitch; % 重新定义像素大小, um

%%  创建文件夹
% 创建指定文件夹路径
folderPath = './TimingData';

HoloDPath= [folderPath , '/HoloD/'];
CADPath= [folderPath , '/CAD/'];
HoloDImgPath= [folderPath , '/HoloD/Img/'];
PhaseDImgPath= [folderPath , '/CAD/Phase/'];
AmpDImgPath= [folderPath , '/CAD/Amp/'];
DepthDImgPath= [folderPath , '/CAD/Depth/'];

HoloSPath= [folderPath , '/HoloS/'];
CASPath= [folderPath , '/CAS/'];
HoloSImgPath= [folderPath , '/HoloS/Img/'];
PhaseSImgPath= [folderPath , '/CAS/Phase/'];
AmpSImgPath= [folderPath , '/CAS/Amp/'];
DepthSImgPath= [folderPath , '/CAS/Depth/'];

HoloNPath= [folderPath , '/HoloN/'];
CANPath= [folderPath , '/CAN/'];
HoloNImgPath= [folderPath , '/HoloN/Img/'];
PhaseNImgPath= [folderPath , '/CAN/Phase/'];
AmpNImgPath= [folderPath , '/CAN/Amp/'];
DepthNImgPath= [folderPath , '/CAN/Depth/'];

    mkdir(folderPath);

    mkdir(HoloDImgPath);
    mkdir(PhaseDImgPath);
    mkdir(AmpDImgPath);
    mkdir(DepthDImgPath);

    mkdir(HoloSImgPath); 
    mkdir(PhaseSImgPath);
    mkdir(AmpSImgPath);
    mkdir(DepthSImgPath);

    mkdir(HoloNImgPath);
    mkdir(PhaseNImgPath);
    mkdir(AmpNImgPath);
    mkdir(DepthNImgPath);

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

HoloS = cell(0); % 静态全息图
HoloN = cell(0); % 全局全息图
HoloD = cell(0); % 动态全息图

CAS = cell(0); % 静态颗粒复振幅信息
CAN = cell(0); % 所有颗粒复振幅信息
CAD = cell(0); % 动态颗粒复振幅信息

%% 添加颗粒samples到ss

 % 从所有模型中随机选取，先添加N1个动态颗粒，再添加N2个静态颗粒
ss = addsamples(ss,N1,'type','mask','R',[1,length(ss.models)],'n',[nr_min,ni_min;nr_max,ni_max]);
ss = addsamples(ss,N2,'type','mask','R',[1,length(ss.models)],'n',[nr_min,ni_min;nr_max,ni_max]);

% 重新指定静态S颗粒位置xy，单位pixel，后面N/2个颗粒静态
% ss.samples.x(floor(n/2)+1:end,:) = [50,200,300,400,500,150,250,350,450,550]; 

% 指定静态颗粒两种极端深度z
ss.samples.z(N1+1:floor(N1+N2/2),:) = ss.backGround.z-(zmax-ss.opticaParams.zcam)/ss.backGround.dz+1; % 不等于0
ss.samples.z(floor(N1+N2/2)+1:end,:) = ss.backGround.z-(zmin-ss.opticaParams.zcam)/ss.backGround.dz; % 切片总数

%% 保存静态颗粒图

Allsamples = ss.samples; % 暂存全部颗粒位置

% 静
ss.samples(1:N1,:) = []; % 仅留静态颗粒位置
ss = forwardV(ss); % 前向传播
[ss,cam1] = forwardCam(ss);
a1 = abs(gather(cam1));
HoloS = a1; % 静态颗粒前向传播
PhaseSMap = zeros(FOV);
AmpSMap = zeros(FOV);
DepthSMap = zeros(FOV);
for j = 1:length(ss.samples.x)
    % CAS 各列分别为 x,y,z,n,PhaseMap, AmpMap, DepthMap. 
    % 单位：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数,um
    CAS{j,1} = ss.samples.x(j);
    CAS{j,2} = ss.samples.y(j);
    CAS{j,3} = (ss.backGround.z-ss.samples.z(j))*ss.backGround.dz+ss.opticaParams.zcam; % 距离相机实际位置，um
    CAS{j,4} = ss.samples.n(j);
    index = ss.models{round(ss.samples.R(j))};

        [CAS{j,5},CAS{j,6},CAS{j,7}] = PhaseSlice(FOV,...
            ss.samples.x(j),...
            ss.samples.y(j),...
            CAS{j,3},... % 距离相机实际位置，um
            index,...
            ss.samples.n(j),...
            n0,lamda);

    PhaseSMap =  PhaseSMap + CAS{j,5}; % 多层叠加相位图，nan
    AmpSMap = AmpSMap + CAS{j,6}; % 多层叠加振幅图，nan
    DepthSMap = DepthSMap + CAS{j,7}; % 多层叠加深度，um
end

AmpSMap = AmpSMap - length(ss.samples.x)*ones(FOV) + ones(FOV); % 光强衰减系数

filename1 = [HoloSPath,num2str(1,'%03d'),'.mat'];
filename2 = [HoloSImgPath,num2str(1,'%03d'),'.png'];
filename3 = [CASPath,num2str(1,'%03d'),'.mat'];
filename4 = [PhaseSImgPath,num2str(1,'%03d'),'.png'];
filename5 = [AmpSImgPath,num2str(1,'%03d'),'.png'];
filename6 = [DepthSImgPath,num2str(1,'%03d'),'.png'];

save(filename1, "HoloS"); %  颗粒全息图，强度图
imwrite(uint8((HoloS/2 * 256)), filename2); %  强度到灰度：/2*256，单位：nan
save(filename3, "CAS","PhaseSMap","AmpSMap","DepthSMap"); %  颗粒所有信息
imwrite(uint8((PhaseSMap)/pi*256), filename4); %  相位图到灰度图：基于pi归一化，单位：nan
imwrite(uint8((AmpSMap)/1*256), filename5); %  振幅图到灰度图：光强衰减系数，单位：nan
imwrite(uint8((DepthSMap)/zmax*256), filename6); %  深度图到灰度图: /zmax*256

ss.samples = Allsamples; % 返回原始全部颗粒位置

%% 指定动态颗粒初始位置

for jj = 1:N1 % 指定动态颗粒位置
    if ss.samples.y(jj)>500
        ss.samples.y(jj) = -999;
    end

    if ss.samples.x(jj)>500
        ss.samples.x(jj) = -999 ;
    end



end

i=(-1)^0.5;

%% 生成时序图
    for t = 1:T % 生成t张时序图
        % 全
        ss = gendf(ss,1);
        ss = forwardV(ss); % 前向传播
        [ss,cam1] = forwardCam(ss);
        HoloN = abs(gather(cam1));

        PhaseNMap = zeros(FOV);
        AmpNMap = zeros(FOV);
        DepthNMap = zeros(FOV);

        for j = 1:length(ss.samples.x)
    % CAN 各列分别为 x,y,z,n,PhaseMap, AmpMap. 
    % 单位：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数
                CAN{j,1} = ss.samples.x(j);
                CAN{j,2} = ss.samples.y(j);
                CAN{j,3} = (ss.backGround.z-ss.samples.z(j))*ss.backGround.dz+ss.opticaParams.zcam; % 距离相机实际位置，um
                CAN{j,4} = ss.samples.n(j);

            index = ss.models{round(ss.samples.R(j))};

            [CAN{j,5},CAN{j,6},CAN{j,7}] = PhaseSlice(FOV,...
                ss.samples.x(j),...
                ss.samples.y(j),...
                CAN{j,3},... % 距离相机实际位置，um
                index,...
                ss.samples.n(j),...
                n0,lamda);
         
            PhaseNMap = PhaseNMap + CAN{j,5}; % 多层叠加相位图，nan
            AmpNMap = AmpNMap + CAN{j,6}; % 多层叠加振幅图，nan
           DepthNMap = DepthNMap + CAN{j,7}; % 多层叠加深度图，um
        end

        AmpNMap = AmpNMap - length(ss.samples.x)*ones(FOV) + ones(FOV); % 光强衰减系数

        filename1 = [HoloNPath,num2str(t,'%03d'),'.mat'];
        filename2 = [HoloNImgPath,num2str(t,'%03d'),'.png'];
        filename3 = [CANPath,num2str(t,'%03d'),'.mat'];
        filename4 = [PhaseNImgPath,num2str(t,'%03d'),'.png'];
        filename5 = [AmpNImgPath,num2str(t,'%03d'),'.png'];
        filename6 = [DepthNImgPath,num2str(t,'%03d'),'.png'];

        save(filename1, "HoloN"); %  颗粒全息图，强度图
        imwrite(uint8((HoloN/2 * 256)), filename2); %  强度到灰度：/2*256，单位：nan
        save(filename3, "CAN","PhaseNMap","AmpNMap","DepthNMap"); %  颗粒所有信息
        imwrite(uint8((PhaseNMap)/pi*256), filename4); %  相位图到灰度图：基于pi归一化，单位：nan
        imwrite(uint8((AmpNMap)/1*256), filename5); %  振幅图到灰度图：光强衰减系数，单位：nan
        imwrite(uint8((DepthNMap)/zmax*256), filename6); %  深度图到灰度图: /zmax*256
                    
        %%
        % 动态
        dd = ss;
        dd.samples(N1+1:end,:) = []; % 仅留动态态颗粒位置
        dd = gendf(dd,1);
        dd = forwardV(dd); % 前向传播
        [dd,cam1] = forwardCam(dd);
        HoloD = abs(gather(cam1)); % 动态颗粒全息图

        PhaseDMap = zeros(FOV);
        AmpDMap = zeros(FOV);
        DepthDMap = zeros(FOV);
  
        for j = 1:length(dd.samples.x)
    % CAD 各列分别为 x,y,z,n,PhaseMap, AmpMap. 
    % 单位：pixel, pixel, um, 绝对折射率, 相对相位,光强衰减系数
                CAD{j,1} = dd.samples.x(j);
                CAD{j,2} = dd.samples.y(j);
                CAD{j,3} = (dd.backGround.z-dd.samples.z(j))*dd.backGround.dz+dd.opticaParams.zcam; % 距离相机实际位置，um
                CAD{j,4} = dd.samples.n(j);

            index = dd.models{round(dd.samples.R(j))};

            [CAD{j,5},CAD{j,6},CAD{j,7}] = PhaseSlice(FOV,...
                ss.samples.x(j),...
                ss.samples.y(j),...
                CAD{j,3},... % 距离相机实际位置，um
                index,...
                ss.samples.n(j),...
                n0,lamda);
         
           
            PhaseDMap = PhaseDMap + CAD{j,5}; % 多层叠加相位图，nan
            AmpDMap = AmpDMap + CAD{j,6}; % 多层叠加振幅图，nan
            DepthDMap = DepthDMap + CAD{j,7}; % 多层叠加深度图，um
        end
        
        AmpDMap = AmpDMap - length(dd.samples.x)*ones(FOV) + ones(FOV); % 光强衰减系数

        filename1 = [HoloDPath,num2str(t,'%03d'),'.mat'];
        filename2 = [HoloDImgPath,num2str(t,'%03d'),'.png'];
        filename3 = [CADPath,num2str(t,'%03d'),'.mat'];
        filename4 = [PhaseDImgPath,num2str(t,'%03d'),'.png'];
        filename5 = [AmpDImgPath,num2str(t,'%03d'),'.png'];
        filename6 = [DepthDImgPath,num2str(t,'%03d'),'.png'];

        save(filename1, "HoloD"); %  颗粒全息图，强度图
        imwrite(uint8((HoloD/2 * 256)), filename2); %  强度到灰度：/2*256，单位：nan
        save(filename3, "CAD","PhaseDMap","AmpDMap","DepthDMap"); %  颗粒所有信息
        imwrite(uint8((PhaseDMap)/pi*256), filename4); %  相位图到灰度图：基于pi归一化，单位：nan
        imwrite(uint8((AmpDMap)/1*256), filename5); %  振幅图到灰度图：光强衰减系数，单位：nan
        imwrite(uint8((DepthDMap)/zmax*256), filename6); %  深度图到灰度图: /zmax*256
                    

        % 运动一帧
        ss.samples.x(1:N1) = ss.samples.x(1:N1)+v(:,1);
        ss.samples.y(1:N1) = ss.samples.y(1:N1)+v(:,2);

    end

