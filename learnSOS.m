function [val,P,xEq] = learnSOS(X,Y,opt)
%%LEARNSOS Returns a pdf matrix P for Sum of Squares(SOS) Lyapunov function
% Given the Data set (X,Y) from a time discrete system x_k+1 = f(x_k)
% Y = x_k+1 , X = x_k , it finds a SOS_P (defined by P) such that the
% violation of the stability condition SOS_P(x_k+1) - SOS_P(x_k)< 0
% is minimized. If val is zero, all data points are stable for the SOS_P
%               P = arg min ramp(sum(SOS_P(x_k+1) - SOS_P(x_k)))
%                   s.t.  all eig(P) > alpha , alpha > 0
% where ramp(x) = 0 for x<0 and ramp(x) = x for x>0.
% The SOS is written  using mon = monomials(x) as SOS_P = mon'*P*mon
% It enforces all Eigenvalues of P to be larger than alpha
% In:
%   X       E  x N     Training data current step
%   Y       E  x N     Training data next step
%   opt.
%        maxP        Lower bound for Eigenvalues of P (default = 1e-8, not used)
%        minP        Lower bound for Eigenvalues of P (default = 1e-5)
%        opt         Options for optimizer fmincon
%        dSOS        degree of the SOS function (default = 2)
% Out:
%   P       Dm  x Dm   pdf matrix for SOS Lyapunov function
%   val     1  x 1     final value of optimization
% N: number of training points
% E: Dimensionality of data
% Dm: dimension of monomial
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

% Fill default value
if ~isfield(opt,'minP'), opt.minP = 1e-4; end
if ~isfield(opt,'maxP'), opt.maxP = 1e8; end
if ~isfield(opt,'dSOS'), opt.dSOS = 2; end
if ~isfield(opt,'opt'), warning('no optimizer options defined');end

% Verfiy Sizes
[E,N] = size(X);
if size(Y,1)~=E || size(Y,2)~=N || ~isscalar(opt.maxP) || ~isscalar(opt.minP) || ~isscalar(opt.dSOS) 
    error('wrong dimension');
end

opt.exMat=getExpoMatrix(E,opt.dSOS);
Em = size(opt.exMat,1);
iL =tril(true(Em)); L0 = randn(Em);

% Define optimization problem
prob.options = opt.opt;
prob.solver = 'fmincon';
if nargout >2
    prob.objective = @(x) fun(x(E+1:end),X,Y,opt,x(1:E));
    prob.x0 =[mean(X,2); L0(iL(:)) ];
    prob.nonlcon = @(x) con(x(1:E),x(E+1:end),opt);

else
    prob.objective = @(L) fun(L,X,Y,opt);
    prob.x0 =L0(iL(:));
    prob.nonlcon = @(x) con(zeros(E,1),x,opt);    
end

prob.objective(prob.x0); %checkGrad(prob.objective,1,1,2,{prob.x0});
prob.nonlcon(prob.x0);  %checkGrad(prob.nonlcon,1,1,3,{prob.x0});

% Solve optimization
[x, val] = fmincon(prob);

L = tril(ones(Em)); 
if nargout > 2
    L(iL(:)) = x(E+1:end); 
    xEq = x(1:E);
else 
    L(iL(:)) = x;
end
P=L*L';


function [f, dfdx] = fun(Lvec,X,Y,opt,xEq)
[E,N,M] = size(Y);
Em = nchoosek(opt.dSOS+E,E)-1; triDm = (Em+1)*Em/2;
if nargin < 5, xEq = zeros(E,1);end

if nargout <= 1
    P = Lvec2SPD(Lvec);
        Vy = zeros(N,M);
        for m = 1:M
                Vy(:,m) = SOS(Y(:,:,m)-xEq,P,opt.exMat);
        end
         Vy = sum(Vy,2)/M;
   
    Vx = SOS(X-xEq,P,opt.exMat);
    dV =  Vy - Vx;
    iinc =  dV > 0;
    f = sum(dV(iinc));
else
    if M>1, error('not supported yet');end
        iNN = 1:N+1:N^2;

    [P, dPdLvec] = Lvec2SPD(Lvec);

    [Vy,dVydx,dVydP] = SOS(Y-xEq,P,opt.exMat);    dVydxEq= -dVydx;
    [Vx,dVxdx,dVxdP] = SOS(X-xEq,P,opt.exMat);    dVxdxEq= -dVxdx;
    dV =  Vy - Vx;  
    iinc =  dV > 0;
    f = sum(dV(iinc));
    dfdLvec  = sum(reshape(dVydP(iinc,:,:)-dVxdP(iinc,:,:),sum(iinc),Em^2)*...
        reshape(dPdLvec,Em^2,triDm),1);
        if nargin < 5
        dfdxEq = []; 
    else
      dfdxEq = permute(dVydxEq-dVxdxEq,[2 1 3]);
    dfdxEq = dfdxEq(:,iNN);
    dfdxEq= sum(dfdxEq(:,iinc),2);
        end
        
     dfdx =  [ dfdxEq' dfdLvec];
    
end



function [c, ceq] = con(xEq,Lvec,opt)
% Compute number of elements and reconstruct L
E = size(xEq,1);
Em = size(opt.exMat,1); %triEm = (Em+1)*Em/2;

L = tril(ones(Em)); L(L==1) = Lvec;

P = L*L';
if nargout <=2
    c = zeros(Em,2);
    % Formulating constraint
    c(1:Em,1) = eig(P) - opt.maxP;
    c(1:Em,2) = -eig(P) + opt.minP;
    c = c(:);
    ceq = [];
else
    warning('no gradient implemented');
end

% if nargout > 2
%     dcdLvec = zeros(Dm,triDm);
%     for tridm =1:triDm
%         for dm = 1:Dm
%             [i,j] =ind2sub([Dm Dm],Lii(tridm));
%             dL = zeros(Dm); dL(i,j) = 1;
%             dcdLvec(dm,tridm) =-Q(:,dm)'*(L*dL'+dL*L')*Q(:,dm);
%         end
%     end
%     dceqdLvec = [];
% end



% function [f, dfdLvec] = fun(Lvec,X,Y,degree, exMat)
% [D,N] = size(X);
% Dm = nchoosek(degree+D,D)-1; triDm = (Dm+1)*Dm/2;
% itri = tril(true(Dm))==true; L = zeros(Dm);
% L(itri(:)) = Lvec; Lii= find(itri);
% f=0;
% dfdLvec = zeros(numel(Lvec),1);
%
% for n=1:N
%     mx = getMonomial(X(:,n),exMat);
%     my = getMonomial(Y(:,n),exMat);
%     val  = max(my'*(L*L')*my - mx'*(L*L')*mx,0);
%     if val > 0
%         f = f + val/N;
%         for tridm =1:triDm
%             [i,j] =ind2sub([Dm Dm],Lii(tridm));
%             dL = zeros(Dm); dL(i,j) = 1;
%             dfdLvec(tridm) =dfdLvec(tridm) + (my'*(L*dL'+dL*L')*my - mx'*(L*dL'+dL*L')*mx)/N ;
%         end
%
%     end
% end
