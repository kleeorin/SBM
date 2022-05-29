function [x,output,exitflag] = sMinimizer(fun,x0,options)

%% Options
if ~isfield(options,'TolX')
    options.TolX=10^-7;     % tolerance change in variables along max direction
end
if ~isfield(options,'TolG')
    options.TolG=10^-7;     % tolerance gradient size along max diection
end
if ~isfield(options,'TolT')
    options.TolT=10^-4;     % tolerance min stepsize
end
if ~isfield(options,'gtdTol')
    options.gtdTol=50;     % tolerance min stepsize
end
if ~isfield(options,'maxCycle')
    options.maxCycle=100;   % stopping criteria of iter without improvement
end
if ~isfield(options,'maxIter')
    options.maxIter=300;    % stopping criteria max interations
end
if ~isfield(options,'skip')
    options.skip=10;        % logging frequency
end
if ~isfield(options,'record')
    options.record=0;        % logging type, 0 means avg only, 1 means record the whole model 
end
if ~isfield(options,'t0')
    options.t0=1;        % usual step, default at 1.
end
% Initialize
h = zeros(length(x0),1);
x = x0;
xmin=x;
tic
% Evaluate Initial Point and lbfgs params
[g] = fun(x);
h = -g;
s = zeros(length(x0),options.m);
y = zeros(length(x0),options.m);
ys = zeros(options.m);
hess_diag = 1;
gtd = -g'*h;
mingtd=gtd;
mini=1;
exitflag=0;
wt=[];
if options.record==1
    wt(end+1,:)=x;
elseif options.record==0
    [jx,hx]=Jw(x,options.q);
    wt(end+1,1)=mean(mean(Frob(jx)));
    wt(end+1,2)=mean(mean(mean(hx.^2)));
end
ind=zeros(1,options.m);ind(1)=1;
skipping=0;
% Output Log
fprintf('%10s   %13s   %12s   %12s %12s\n','Iteration','Step Length','X change','Gradient','gtd');

%% Loop until break condition
for i = 1:options.maxIter


    if i>1
        t=options.t0;
    else
        t = min(1,1/sqrt(sum(g.^2)));
    end
    g_old=g;
    h_old=h;
    gtd_old=gtd;
    [x,t,h,g,gtd,s,y,ys,hess_diag,ind,skipping]=advanceSearch(x,t,h,g,fun,gtd,s,y,ys,hess_diag,i,options,ind,skipping);
    if mingtd>gtd
        mingtd=gtd;
        mini=i;
        %xmin=x;
    end
    fprintf('%10d     %3.6e    %3.6e    %3.6e     %3.6e\n',i,t,max(abs(t*h_old)),max(abs(g_old)),gtd_old);

    
    output.step(i)=t;
    output.Xchange(i)=max(abs(t*h_old));
    output.grad(i)=sum((g_old).^2);
    output.gtd(i)=gtd_old;
    output.gg(i)=dot(g,g_old)/sqrt(dot(g,g))/sqrt(dot(g_old,g_old));
    output.timer(i)=toc;
%%  Break Conditions

    if max(abs(t*h)) <= options.TolX
        exitflag=1;
        msg = 'X Change below TolX';
    end
    if max(abs(g)) <= options.TolG
        exitflag=1;
        msg = 'gradient Change below TolG';
    end
    
    if i == options.maxIter
        exitflag = 2;
        msg='Reached Maximum Number of Iterations';
    end
    
    if t < options.TolT
        exitflag = 3;
        msg='Step size below TolT';
    end
    
    if i-mini>options.maxCycle && i>15+options.maxCycle
        exitflag = 3;
        msg=['maxCycle=' num2str(options.maxCycle) ' without improvement'];
    end

    if exitflag>0
        output.exitflag=exitflag;
        break;    
    end
    if mod(i,options.skip)==0
        if options.record==1
            wt(end+1,:)=x;
        elseif options.record==0
            [jx,hx]=Jw(x,options.q);
            wt(end+1,1)=mean(mean(Frob(jx)));
            wt(end,2)=mean(mean(mean(hx.^2)));
        end
    end
    
    % log output
end
%x=xmin;
fprintf('%s\n',msg);
output.msg=msg;
output.skipping=skipping;
output.wt=wt;
end

