classdef forwardAng < simulateSpaceSphere
    %FORWARDANG - 使用角谱计算三维场全息图 
    %
    %   forwardAng类使用角谱计算仿真三维球形粒子场的全息图。你可以通过设置系统的
    %   光学参数改变计算的全息图。通过forwardV可以计算所记录的三维球形粒子场的波
    %   前，通过forwardZ可以计算对应平面的传播结果，通过forwardCam可以计算对应
    %   波前传播至相机的结果
    %
    %   simulateSpace properties:
    %       *backGround     - parameters of simulate space
    %       *samples        - table of simulate samples
    %       opticaParams    - parameters of optical system
    %
    %   simulateSpace methods:
    %       *addmodels      - add models to cell
    %       *addsamples     - add samples to table
    %       *resamples      - remove samples from table
    %       *mkvol          - generate voxel from simulateSpace
    %       *mkvolSingle    - generate voxel from single sample
    %       gendf           - generate initial light field
    %       forwardV        - forward through simulate voxel
    %       forwardCam      - forward to camera
    %       forwardZ        - forward through plane
    %       mask            - caculate phase in plane
    %       gentf           - caculate angular spectrum transfer
    %       padding         - padding plane
    %
    %   Example:
    %       ss = forwardAng; % 构建类
    %       ss = addsamples(ss,20,'r',5); % 添加20个半径为5的颗粒
    %       ss = forwardV(ss); % 前向传播
    %       [~,u_cam] = forwardCam(ss); % 传播至相机
    %       abs(u_cam); % 强度图
    %
    %   See also simulateSpaceSphere.

    %   MATLAB22b - 2023.10.30 - by SZU-IPC
    properties
        % opticaParams  光学参数
        opticaParams
        % forwardData   前向传播数据
        forwardData
    end
    methods
        function this = forwardAng(par)
            %FORWARDANG - 构建forwardAng类

            arguments
                par.lamda = 0.532 % 波长 (um)
                par.zcam = 1000 % 到相机距离(um)
                par.n0 = 1.33 % 背景折射率
                par.padlen = 1 % 原始对象1倍
                par.outip = 0; % 输出显示
                par.xx = 0; % 非相干叠加
            end
            this.opticaParams.lamda = par.lamda;
            this.opticaParams.zcam = par.zcam;
            this.opticaParams.n0 = par.n0;
            this.opticaParams.padlen = par.padlen;
            this.opticaParams.nx = (2*par.padlen+1)*this.backGround.xy(1);
            this.opticaParams.ny = (2*par.padlen+1)*this.backGround.xy(2);
            this.backGround.dz = par.lamda/(4*par.n0);
            this.opticaParams.outip = par.outip;
            this.opticaParams.xx = par.xx;
            % 前向传播数据
            this.forwardData.u_in = [];
            this.forwardData.u_g = [];
            this.forwardData.u_cam = [];
            this = gendf(this,1);
        end

        function this = gendf(this,in_field_map)
            %GENDF - 生成初始的输入场
            %   此函数返回forwardAng类，并向其中添加(修改)入射场
            %
            %   forwardAng = GENDF(forwardAng,in_field_map[1]) 使用单元素数值
            %                in_field_map指定入射场，将设定振幅为in_field_map
            %                的平面波入射场
            %   forwardAng = GENDF(forwardAng,in_field_map[nx,ny])  使用虚数
            %                矩阵in_field_map指定自定义入射场

            this.forwardData.u_in = in_field_map .* ...
                ones(this.opticaParams.nx, this.opticaParams.ny);
            this.forwardData.u_g = in_field_map .* ...
                ones(this.opticaParams.nx, this.opticaParams.ny);
            if canUseGPU()
                this.forwardData.u_in = gpuArray(this.forwardData.u_in);
                this.forwardData.u_g = gpuArray(this.forwardData.u_g);
            end
        end

        function [this,u_g] = forwardV(this,zrange,u_in)
            %FORWARDV - 前向传播通过仿真体
            %   此函数返回forwardAng类与传播后的波前，并在类中记录传播场
            %
            %   forwardAng = forwardV(forwardAng)   以记录的输入场通过仿真空间
            %                并将波前记录在forwardAng类中
            %   [forwardAng,u_g] = forwardV(forwardAng) 以记录的输入场通过仿
            %                      真空间并将波前记录在forwardAng类中,同时以变
            %                      量u_g输出
            %   [forwardAng,u_g] = forwardV(...,zange,u_in) 自定义计算体空间
            %                      范围zange与输入场u_in  

            arguments
                this
                zrange = []
                u_in = []
            end
            if isempty(zrange)
                zrange = [1,this.backGround.z];
            end
            if isempty(u_in)
                if this.opticaParams.xx
                    this.forwardData.u_g = abs(this.forwardData.u_g); % 去除相位
                end
                u_in = this.forwardData.u_g;
                this.forwardData.u_in = u_in;
            end 
            samples = this.samples(this.samples.z>=zrange(1)&this.samples.z<=zrange(2),:);
            z = sort(unique(samples.z)); % 样本排序
            dz = [diff(z);this.backGround.z-z(end)]; % 样本间距
            for i = 1:numel(z)
                if this.opticaParams.outip
                    outputTip('tip','Depth','N',z(i))
                end
                s = samples(samples.z==z(i),:);
                switch s.type{1}
                    case 'sphere'
                        I = mask(this,s); % 计算平面相位
                    case 'mask'
                        I = maskmodel(this,s); % 计算平面相位
                end
                [this,u_g] = forwardZ(this,this.backGround.dz*dz(i),I); % 前向传播
                
            end
        end

        function [this,u_cam] = forwardCam(this)
            %FORWARDCAM - 前向传播至相机
            %   此函数返回forwardAng类与传播至相机后的波前，并在类中记录传播场
            %
            %   forwardAng = forwardCam(forwardAng) 以记录的输出场传播至相机
            %                并将波前记录在forwardAng类中
            %   [forwardAng,u_cam] = forwardCam(forwardAng) 以记录的输入场传
            %                        播至相机并将波前记录在forwardAng类中,同时
            %                        以变量u_cam输出

            [~,u_cam] = forwardZ(this,this.opticaParams.zcam);
            u_cam = u_cam(this.opticaParams.padlen*this.backGround.xy(1)+1:...
                (this.opticaParams.padlen+1)*this.backGround.xy(1),...
                this.opticaParams.padlen*this.backGround.xy(2)+1:...
                (this.opticaParams.padlen+1)*this.backGround.xy(2));
            this.forwardData.u_cam = u_cam;
        end

        function [this,u_g] = forwardZ(this,z,I,u_in)
            %FORWARDZ - 前向传播通过相位面
            %   此函数返回forwardAng类与传播后的波前，并在类中记录传播场
            %
            %   forwardAng = forwardZ(forwardAng)   以记录的输入场通过相位面
            %                并将波前记录在forwardAng类中
            %   [forwardAng,u_g] = forwardV(forwardAng) 以记录的输入场通过相
            %                      位面并将波前记录在forwardAng类中,同时以变
            %                      量u_g输出
            %   [forwardAng,u_g] = forwardV(...,z,I,u_in) 自定义传播距离z,相
            %                      位面I与输入场u_in  

            arguments
                this
                z
                I = []
                u_in = []
            end
            if isempty(I)
                I = ones(this.backGround.xy(1),this.backGround.xy(2));
            end
            if isempty(u_in)
                if this.opticaParams.xx
                    this.forwardData.u_g = abs(this.forwardData.u_g); % 去除相位
                end
                u_in = this.forwardData.u_g;
                this.forwardData.u_in = u_in;
            end
            I = padding(this,I); % 放大
            prop = gentf(this,z,I); % 传递函数
            prop_fft = fftshift(fft2(fftshift(I.*u_in)));
            u_g = fftshift(ifft2(fftshift(prop_fft.*prop)));
            this.forwardData.u_g = u_g;
        end

        function I = mask(this,s)
            %MASK - 计算球形颗粒相位面
            %   此函数返回相位面I
            %
            %   I = mask(forwardAng,s)  计算s表格中的样本在forwardAng参数下的
            %       相位面

            [x,y] = meshgrid(1:this.backGround.xy(1),1:this.backGround.xy(2));
            for i = 1:height(s)
                D = (this.backGround.dxy*(s.x(i)-x)).^2+...
                    (this.backGround.dxy*(s.y(i)-y)).^2;
                D(D>s.R(i)^2) = s.R(i)^2;
                phD = 2*sqrt(s.R(i)^2-D);
                dn = s.n(i)-this.opticaParams.n0;
                phI{i} = this.backGround.dz*phD*2*pi*dn/this.opticaParams.lamda;
            end
            I = sum(cat(3,phI{:}),3);
            I = exp(-1i*I);
        end
        
        function I = maskmodel(this,s)
            %MASKMODEL - 计算模板颗粒相位面
            %   此函数返回相位面I
            %
            %   I = maskmodel(forwardAng,s)  计算s表格中的样本在forwardAng参
            %       数下的相位面

            d = 10; % 厚度, um
            for i = 1:height(s)
                phI{i} = zeros(this.backGround.xy(1),this.backGround.xy(2));
                wx = this.models{s.R(i)}(:,1)+s.x(i);              
                w1 = find(wx<1|wx>this.backGround.xy(1));
                wy = this.models{s.R(i)}(:,2)+s.y(i);
                w2 = find(wy<1|wy>this.backGround.xy(2));
                wd = this.models{s.R(i)}(:,3);
                wk = unique([w1;w2]);
                wx(wk) = [];
                wy(wk) = [];
                wd(wk) = [];
                dn = s.n(i)-this.opticaParams.n0;
                k = 2*pi*dn/this.opticaParams.lamda;
                for j = 1:numel(wx)
                    phI{i}(wx(j),wy(j)) = k*wd(j);
                end
            end
            I = sum(cat(3,phI{:}),3);
            I = exp(-1i*I);
        end

        function prop = gentf(this,z,I)
            %GENTF - 计算角谱传递函数
            %   此函数返回角谱传递函数prop
            %
            %   prop = gentf(forwardAng,z,I)    计算forwardAng类参数下，相位
            %          面I传播距离z时的角谱传递函数

            [M,N] = size(I);
            pixelnumx = N;
            pixelnumy = M;
            deltafx = 1/M/this.backGround.dxy;
            deltafy = 1/N/this.backGround.dxy;
            Lx = pixelnumx*this.backGround.dxy;
            Ly = pixelnumy*this.backGround.dxy;
            xarray = 1:pixelnumx;
            yarray = 1:pixelnumy;
            [xgrid,ygrid] = meshgrid(xarray,yarray);
            prop = exp(2*1i*pi*this.opticaParams.n0*abs(z)*...
                ((1/this.opticaParams.lamda)^2-...
                ((xgrid-pixelnumx/2-1).*deltafx).^2-...
                ((ygrid-pixelnumy/2-1).*deltafy).^2).^0.5);
            if canUseGPU
                prop = gpuArray(prop);
            end
        end

        function I = padding(this,I,pad)
            %PADDING - 填充图片
            %   返回填充后的图片
            %
            %   I = padding(forwardAng,I)   按forwardAng类参数填充I
            %   I = padding(forwardAng,I,pad)   按参数pad填充I

            arguments
                this
                I
                pad = [];
            end
            if isempty(pad)
                pad = this.opticaParams.padlen;
            end
            if pad ~= 0
                [M,N] = size(I);
               I = padarray(I,[pad*M,pad*N],1,'both'); 
            end
        end
    end
end