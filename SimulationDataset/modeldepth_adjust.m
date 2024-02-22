function I = modeldepth_adjust(K,F)
    arguments
        K
        F = @adfun_exp
    end
    K = K~=0;
    D = nan(size(K));
    r = 0;
    while any(K,'all')
        r = r+1;
        T = imerode(K,[0,1,0;1,1,1;0,1,0]);
        D(T~=K) = r;
        K = T;
    end
    R = max(D,[],'all'); % 最大半径(pixel)
    D = R-D;
    D(~isnan(D)) = F(D(~isnan(D)),R);
    I = fillmissing(D,"constant",0);
end

function D = adfun_exp(r,R,maxdep_pp,mindep)
    
    arguments
        r
        R
        maxdep_pp = 20 % 最大深度，单位：um
        mindep = 1 % 最小深度，单位：um
    end
    global  maxdep_pp
    k = log(maxdep_pp+1-mindep)/R; 
    D = (maxdep_pp+1)-exp(k*r); % 中心处为最大深度，边缘为最小深度
end