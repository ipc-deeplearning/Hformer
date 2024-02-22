% 输入参数视场、颗粒中心位置、mask'index、折射率、波长
% 单位分别为：pixel,pixel,pixel,um,pixel,nan,nan,um，其中z为实际距离
% 输出叠加的相位图、叠加的振幅图、叠加的深度图
function [Phase,Amp,Depth]= PhaseSlice(FOV,x,y,z,index,n,n0,lamda)
    x = floor(x);
    y= floor(y);

    X = index(:,1)-min(index(:,1))+1+x;
    Y = index(:,2)-min(index(:,2))+1+y;
    D = zeros(FOV-3*min(index(:,1)),FOV-3*min(index(:,2)));
    Depth = zeros(FOV-3*min(index(:,1)),FOV-3*min(index(:,2)));
    Phase = zeros(FOV-3*min(index(:,1)),FOV-3*min(index(:,2)));
    Amp = ones(FOV-3*min(index(:,1)),FOV-3*min(index(:,2)));
    for i = 1:height(index)
        if X(i)<=0 || Y(i)<=0
            continue;
        end


        D(X(i),Y(i)) = index(i,3); % 厚度，单位：um
        Phase(X(i),Y(i))= D(X(i),Y(i))*real(n-n0)*2*pi/lamda; % 相位差，单位：rad
        Amp(X(i),Y(i)) = exp((D(X(i),Y(i))*imag(n-n0)*4*pi/lamda)); % 强度衰减系数
        Depth(X(i),Y(i))= D(X(i),Y(i))/D(X(i),Y(i))*z; % 深度，单位：um，z 为实际距离
    end
    Phase = Phase (-min(index(:,1))+2:FOV-min(index(:,1))+1,-min(index(:,2))+2:FOV-min(index(:,2))+1);
    Amp = Amp (-min(index(:,1))+2:FOV-min(index(:,1))+1,-min(index(:,2))+2:FOV-min(index(:,2))+1);
    Depth = Depth (-min(index(:,1))+2:FOV-min(index(:,1))+1,-min(index(:,2))+2:FOV-min(index(:,2))+1);
end
