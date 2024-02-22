clear,clc
ss = forwardAng;
I = imread("2.png");
ss = addmodels(ss,I); % 添加模型
ss = addsamples(ss,100,'type','mask','r',[1,1],'n',[1.4,0.001;1.6,0.01]);
% 添加样本，type->样本种类，r->模型序号，n->折射率
[ss,ug1] = forwardV(ss); % 前向传播
[ss,cam1] = forwardCam(ss);
a1 = abs(gather(cam1));
ss = gendf(ss,1);
ss.opticaParams.xx = 1;
[ss,ug2] = forwardV(ss); % 前向传播
[ss,cam2] = forwardCam(ss);
a2 = abs(gather(cam2));