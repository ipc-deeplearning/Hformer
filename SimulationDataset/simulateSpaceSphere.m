classdef simulateSpaceSphere
    %SIMULATESPACESPHERE - 三维球形粒子场仿真信息
    %
    %   simulateSpaceSphere类用来描述所需仿真的三维球形粒子场。你可以通过设置
    %   仿真空间的信息与仿真样本，改变所需仿真的粒子场。最终通过mkvol函数可以根据
    %   设定的粒子场信息生成对应的体素空间用于BMP计算。
    %
    %   simulateSpace properties:
    %       backGround      - parameters of simulate space
    %       samples         - table of simulate samples
    %       models          - models of samples
    %
    %   simulateSpace methods:
    %       addmodels       - add models to cell
    %       addsamples      - add samples to table
    %       resamples       - remove samples from table
    %       mkvol           - generate voxel from simulateSpace
    %       mkvolSingle     - generate voxel from single sample
    %   See also .

    %   MATLAB22b - 2023.10.16 - by SZU-IPC

    properties
        % backGround    仿真空间信息
        backGround
        % samples       仿真样本列表
        samples
        % models        仿真模型
        models
    end

     methods
        function this = simulateSpaceSphere(par)
            %SIMULATESPACESPHERE - 构建simulateSpaceSphere类
            
            arguments
                par.xy = [1024,1024]; % 横向大小(pixel)
                par.z = 5000; % 纵向深度(pixel)
                par.dxy = 1.85; % 横向精度(um)
                par.lamda = 0.532; % 波长(um)
                par.n = 1.33; % 背景折射率
            end
            % parameters of background
            this.backGround.xy = par.xy;
            this.backGround.z = par.z;
            this.backGround.dxy = par.dxy;
            this.backGround.dz = par.lamda/(4*par.n);
            % parameters of samples
            this.samples = table('Size',[0,6],'VariableTypes',...
                {'double','double','double','double','double','char'},...
                'VariableNames',{'x','y','z','R','n','type'});
            % models of samples
            this.models = cell(0);
        end

        function this = addmodels(this,I)
            %ADDMODELS - 向模型库中添加新模型
            %   此函数返回simulateSpace类，并向模型库中添加模型
            %
            %   simulateSpace = ADDMODELS(simulateSpace,I) 将模型模板I添加进
            %                   模型库中

            k = numel(this.models);
            [m,n] = size(I);
            [row,col] = find(I~=0);
            this.models{k+1} = [row-floor(m/2),col-floor(n/2)];
            this.models{k+1}(:,3) = I(I~=0);
            
        end

        function this = addsamples(this,N,par)
            %ADDSAMPLES - 向类中添加样本
            %   此函数返回simulateSpaceSphere类，并向类的样本表中添加样本
            %
            %   simulateSpace = ADDSAMPLES(simulateSpace,N) 以默认参数向类中
            %                   添加N个样本
            %   simulateSpace = ADDSAMPLES(simulateSpace,...,'Name','Value')
            %                   以自定义参数向类中添加样本
            %
            %   名称-值参数
            %       l - 位置
            %           'rand' | double(N,3)
            %       r - 大小/类型
            %           double(1) | double(1,2) | double(N,1)
            %       n - 折射率
            %           double(1,1) | double(2,1) | double(N,1)
            %           double(1,2) | double(2,2) | double(N,1)
            %       type - 输入类型
            %           'sphere' | 'mask'

            arguments
                this
                N = 15; % 样本个数
                par.l = 'rand'; % 样本位置
                par.r = 10; % 样本大小(um)
                par.n = 1.59;
                par.type = 'sphere'; % 样本类型
            end
            k = height(this.samples);
            % location of samples
            if strcmp(par.l,'rand')
                this.samples.x(k+1:k+N) = unidrnd(this.backGround.xy(1),N,1);
                this.samples.y(k+1:k+N) = unidrnd(this.backGround.xy(2),N,1);
                this.samples.z(k+1:k+N) = unidrnd(this.backGround.z,N,1);
            else
                this.samples.x(k+1:k+N) = par.l(:,1);
                this.samples.y(k+1:k+N) = par.l(:,2);
                this.samples.z(k+1:k+N) = par.l(:,3);
            end
            % size/type of samples   
            if numel(par.r) == 1
                this.samples.R(k+1:k+N,1) = par.r;
            elseif width(par.r) == 2
                this.samples.R(k+1:k+N,1) = par.r(1)+(par.r(2)-par.r(1))*rand(N,1);
            else
                this.samples.R(k+1:k+N,1) = par.r;
            end 
            % refractive index of samples
            if width(par.n) == 1
                if height(par.n) == 1
                    this.samples.n(k+1:k+N,1) = par.n;
                elseif height(par.n) == 2
                    this.samples.n(k+1:k+N,1) = par.n(1)+(par.n(2)-par.n(1))*rand(N,1);
                else
                    this.samples.n(k+1:k+N,1) = par.n;
                end
            else
                if height(par.n) == 1
                    this.samples.n(k+1:k+N,1) = par.n(1)-1i*par.n(2);
                elseif height(par.n) == 2
                    temp1 = par.n(1,1)+(par.n(2,1)-par.n(1,1))*rand(N,1); % 实部
                    temp2 = par.n(1,2)+(par.n(2,2)-par.n(1,2))*rand(N,1); % 虚部
                    this.samples.n(k+1:k+N,1) = temp1-1i*temp2;
                else
                    this.samples.n(k+1:k+N,1) = par.n;
                end
            end
            % type of samples
            for i = 1:N
                this.samples.type{k+i} = par.type;
                if strcmp(par.type,'mask')
                    this.samples.R(k+i) = round(this.samples.R(k+i));
                end
            end
        end

        function this = resamples(this,par)
            %RESAMPLES - 从类中移除部分样本
            %   此函数返回simulateSpace类，并从类的样本表中移除样本
            %
            %   simulateSpace = RESAMPLES(simulateSpace,'N',N) 从类中移除N个样本
            %   simulateSpace = RESAMPLES(simulateSpace,'w',w) 从类中移除第w个样本
            %   simulateSpace = RESAMPLES(simulateSpace) 从类中移除所有样本

            arguments
                this
                par.N = []; % 样本个数
                par.w = []; % 样本序号
            end
            if ~isempty(par.w)
                this.samples(par.w,:) = [];
            elseif ~isempty(par.N)
                w = [];
                while numel(w)<par.N
                    w = [w,unidrnd(height(this.samples),par.N-numel(w),1)];
                    w = unique(w);
                end
                this.samples(w,:) = [];
            else
                this.samples(1:height(this.samples),:) = [];
            end
        end

        function vol = mkvol(this,zrange,k)
            %MKVOL - 由类构建仿真体素空间
            %   此函数返回体素，该体素由simulateSpace类中参数仿真而来
            %
            %   vol = MKVOL(this,zrange)    构建深度范围zrange内的体素空间
            %   vol = MKVOL(this,zrange,k)  构建深度范围zrange内的k样本体素空间

            arguments
                this
                zrange = [1,500]; % 深度范围(pixel)
                k = [];
            end
            Vol = zeros(this.backGround.xy(1),this.backGround.xy(2),...
                zrange(2)-zrange(1)+1);
            [X,Y,Z] = meshgrid(1:this.backGround.xy(1),...
                1:this.backGround.xy(2),zrange(1):zrange(2));
            if isempty(k)
                samples = this.samples;
            else
                samples = this.samples(k,:);
            end
            w = (samples.z - samples.R/this.backGround.dz)<zrange(2) &...
                (samples.z + samples.R/this.backGround.dz)>zrange(1);
            samples = samples(w,:);
            for i = 1:height(samples)
                outputTip('tip','Mkvol','N',i,'O',height(samples))
                Xt = (X - samples.x(i))*this.backGround.dxy;
                Yt = (Y - samples.y(i))*this.backGround.dxy;
                Zt = (Z - samples.z(i))*this.backGround.dz;
                Dt = Xt.^2+Yt.^2+Zt.^2;
                Vol(Dt<=(samples.R(i)^2)) = 1;
            end
            vol = Vol;
        end

        function [vol,z] = mkvolSingle(this,k)
            %MKVOLSINGLE - 构建单个颗粒的体素空间

            samples = this.samples(k,:);
            zrange = [samples.z-samples.R/this.backGround.dz,...
                samples.z+samples.R/this.backGround.dz];
            [X,Y,Z] = meshgrid(1:this.backGround.xy(1),...
                1:this.backGround.xy(2),zrange(1):zrange(2));
            vol = zeros(this.backGround.xy(1),this.backGround.xy(2),...
                zrange(2)-zrange(1)+1);
            Xt = (X - samples.x)*this.backGround.dxy;
            Yt = (Y - samples.y)*this.backGround.dxy;
            Zt = (Z - samples.z)*this.backGround.dz;
            Dt = Xt.^2+Yt.^2+Zt.^2;
            vol(Dt<=(samples.R^2)) = 1;
            z = (this.backGround.z-samples.z)*this.backGround.dz-samples.R;
        end    

     end
end