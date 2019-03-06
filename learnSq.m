function [val,P, xEq] = learnSq(X,Y, opt)
%%LEARNP Returns a spd matrix P for Lyapunov function V(x_k) = x_k'P x_k
% Given the Data set (X,Y) from a time discrete system x_k+1 = f(x_k)
% Y = x_k+1 , X = x_k , it finds a P such that the violation of the
% stability condition x_k+1' P x_k+1 - x_k P x_k < 0 is minimized.
% If val is zero, all data points are stable for P
%               P = arg min ramp(sum(x_k+1'*P*x_k+1' - x_k'*P*x_k'))
%                   s.t.  all eig(P) > alpha , alpha > 0
% where ramp(x) = 0 for x<0 and ramp(x) = x for x>0.
% In:
%     X      E  x N       Training data current step
%     Y      E  x N (x M) Training data next step
%     opt.
%         maxP           Lower bound for Eigenvalues of P (default = 1e-8, not used)
%         minP           Lower bound for Eigenvalues of P (default = 1e-5)
%         opt            Options for optimizer fmincon
% Out:
%    P       E  x E     pdf matrx for Lyapunov function
%    xEq     E  x 1     equilibrium point
%    val     1  x 1     final value of optimization
% N: number of training points
% E: Dimensionality of data
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

% Fill default value
if ~isfield(opt,'minP'), opt.minP = 1e-4; end
if ~isfield(opt,'maxP'), opt.maxP = 1e8; end
if ~isfield(opt,'opt'), warning('no optimizer options defined');end

% Verfiy Sizes
[E,N] = size(X);
if size(Y,1)~=E || size(Y,2)~=N || ~isscalar(opt.minP) || ~isscalar(opt.maxP)
    error('wrong dimension');
end

iL = tril(true(E)); L0 = eye(E);

% Define optimization problem
prob.options = opt.opt;
prob.solver = 'fmincon';
if nargout >2
    prob.objective = @(x) fun(x(E+1:end),X,Y,x(1:E));
    prob.x0 =[mean(X,2); L0(iL(:)) ];
    prob.nonlcon = @(x) con(x(1:E),x(E+1:end),opt);

else
    prob.objective = @(L) fun(L,X,Y);
    prob.x0 =L0(iL(:));
    prob.nonlcon = @(x) con(zeros(E,1),x,opt);    
end
% prob.objective(prob.x0); checkGrad(prob.objective,1,1,2,{prob.x0});
% prob.nonlcon(prob.x0); checkGrad(prob.nonlcon,1,1,3,{prob.x0});


% Solve optimization
[x, val] = fmincon(prob);

% Reconstruct Output
if nargout > 2
    L = tril(ones(E)); L(iL(:)) = x(E+1:end); P=L*L';
    xEq = x(1:E);
else
    L = tril(ones(E));L(iL(:)) = x;P=L*L';
end
end

function [f, dfdx] = fun(Lvec,X,Y,xEq)
[E,N,M] = size(Y); triD = (E+1)*E/2;
if nargin < 4, xEq = zeros(E,1);end
if nargout <=  1
    P = Lvec2SPD(Lvec);
    Vy = zeros(N,M);
    for m = 1:M
        Vy(:,m) = Sq(Y(:,:,m)-xEq,P);
    end
    Vy = sum(Vy,2)/M;
    Vx = Sq(X-xEq,P);
    dV =  Vy - Vx; iinc =  dV > 0;
    f = sum(dV(iinc));
else
    iNN = 1:N+1:N^2;
    [P, dPdLvec] = Lvec2SPD(Lvec);
    Vy = zeros(N,M); dVydP = zeros(N,E,E,M);
    for m = 1:M
        [Vy(:,m),dVydxEq,dVydP(:,:,:,m)] = Sq(Y(:,:,m)-xEq,P);
        dVydxEq= -dVydxEq;
    end
    Vy = mean(Vy,2);
    dVydP = mean(dVydP,4);
    [Vx,dVxdxEq,dVxdP] = Sq(X-xEq,P);
    dVxdxEq = -dVxdxEq;
    dV =  Vy - Vx;
    iinc =  dV > 0;
    f = sum(dV(iinc));
    dfdLvec  = sum(reshape(dVydP(iinc,:,:)-dVxdP(iinc,:,:),sum(iinc),E^2)*...
        reshape(dPdLvec,E^2,triD),1);
    if nargin < 4
        dfdxEq = []; 
    else
        dfdxEq = permute(dVydxEq-dVxdxEq,[2 1 3]);
        dfdxEq = dfdxEq(:,iNN);
        dfdxEq= sum(dfdxEq(:,iinc),2);
    end
    dfdx =  [ dfdxEq' dfdLvec];
    %         dfdx =  dfdxEq;
    
end
end


function [c, ceq] = con(xEq,Lvec,opt)
% Compute number of elements and reconstruct L
E = size(xEq,1);
L = tril(ones(E)); L(L==1) = Lvec;

P = L*L';
if nargout <=2
    c = zeros(E,2);
    % Formulating constraint
    c(1:E,1) = eig(P) - opt.maxP;
    c(1:E,2) = -eig(P) + opt.minP;
    c = c(:);
    ceq = [];
else
    warning('no gradient implemented');
end

% if nargout > 2
%     dcdLvec = zeros(Dm,triD);
%     for tridm =1:triD
%         for dm = 1:Dm
%             [i,j] =ind2sub([Dm Dm],Lii(tridm));
%             dL = zeros(Dm); dL(i,j) = 1;
%             dcdLvec(dm,tridm) =-Q(:,dm)'*(L*dL'+dL*L')*Q(:,dm);
%         end
%     end
%     dceqdLvec = [];
% end

end