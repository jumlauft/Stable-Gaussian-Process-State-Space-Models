function [val,P0,P,mu,xEq] = learnWSAQF(X,Y,opt)
%%LEARNWSAQF Returns a parameters for a WSAQF Lyapunov function
% Given the Data set (X,Y) from a time discrete system x_k+1 = f(x_k)
% Y = x_k+1 , X = x_k , it finds a Weighted Sum of Asymmetric Quadratic
% Functions (WSAQF) (defined by P0,P,mu) such that the
% violation of the stability condition WSAQF(x_k+1) - WSAQF(x_k)< 0
% is minimized. If val is zero, all data points are stable for the WSAQF
%               P = arg min ramp(sum(WSAQF(x_k+1) - WSAQF(x_k)))
% where ramp(x) = 0 for x<0 and ramp(x) = x for x>0.
% For details regarinding WSAQF refer to
%   Khansari-Zadeh, S. M. & Billard, A.
%   Learning control Lyapunov function to ensure stability of dynamical
%   system-based robot reaching motions Robotics and Autonomous Systems,
%   Elsevier, 2014, 62, 752-765
% In:
%     X      E  x N     Input locations
%     Y      E  x N     Output locations
%     opt.
%         maxP        Lower bound for Eigenvalues of P (default = 1e-8)
%         minP        Lower bound for Eigenvalues of P (default = 1e-5, not used)
%         opt         Options for optimizer fmincon
%         dWSAQF      degree of the WSAQF function (default = 2)
% Out:
%     val    1  x 1       final value of optimization
%     P0     E x E       Symmetric square matrix
%     P      E x E x L   Asymmetric square matrices
%     mu     E x L       Asymmetric shift
%     xEq    E x 1       estimated equilibrium point
% E: Dimensionality of data
% N: Number of trianing points
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

% Fill default value
if ~isfield(opt,'minP'), opt.minP = 1e-4; end
if ~isfield(opt,'maxP'), opt.maxP = 1e8; end
if ~isfield(opt,'dWSAQF'), opt.dWSAQF = 2; end
if ~isfield(opt,'opt'), warning('no optimizer options defined');end


% Verfiy Sizes
[E,N] = size(X); triE = E*(E+1)/2;
if size(Y,1)~=E || size(Y,2)~=N || ~isscalar(opt.minP) || ~isscalar(opt.maxP) || ~isscalar(opt.dWSAQF)
    error('wrong dimension');
end
L = opt.dWSAQF;



% Define optimization problem
prob.options = opt.opt;
prob.solver = 'fmincon';
% prob.objective = @(p) fun(p,L,X,Y);
if nargout > 4
    prob.objective = @(x) fun(x(E+1:end),X,Y,opt,x(1:E));
    prob.x0 =[mean(X,2);0.5+rand((L+1)*triE,1); randn(L*E,1)];
    prob.nonlcon = @(x) con(x(1:E),x(E+1:end),opt);
    
else
    prob.objective = @(p) fun(p,X,Y,opt);
    prob.x0 =[0.5+rand((L+1)*triE,1); randn(L*E,1)];
    prob.nonlcon = @(x) con(zeros(E,1),x,opt);
    
end
% prob.objective(prob.x0); checkGrad(prob.objective,1,1,2,{prob.x0});
% prob.nonlcon(prob.x0); checkGrad(prob.nonlcon,1,1,3,{prob.x0});



% Solve optimization
[x,val] = fmincon(prob);

% Prepare output
if nargout > 4
    [P0,P,mu] = p2WSAQF(x(E+1:end),E,L);
    xEq = x(1:E);
else
    [P0,P,mu] = p2WSAQF(x,E,L);
end

end

function [f, dfdx] = fun(p,X,Y,opt,xEq)
% In:
%    p     nP x 1     Paramtervector
%    L     1 x 1      Number of Asymmetric components
%    X     E  x N     Input locations
%    Y     E  x N     Output locations
% Out:
%   f      1  x 1     objective value
%   dfdp   np x 1     derivative

[E,N,M] = size(Y);    if M>1, error('not supported yet');end
L = opt.dWSAQF;
triD = E*(E+1)/2; Np = (L+1)*triD + L*E;
if nargin < 5, xEq = zeros(E,1);end



if nargout <= 1
    [P0,P,mu]  = p2WSAQF(p,E,L);
    Vy = WSAQF(Y-xEq,P0,P,mu);
    Vx = WSAQF(X-xEq,P0,P,mu);
    dV =  Vy - Vx;  iinc =  dV > 0;
    f = sum(dV(iinc));
    
else
    iNN = 1:N+1:N^2;
    
    [P0,P,mu, dP0dp, dPdp, dmudp]  = p2WSAQF(p,E,L);
    
    [Vy,dVydx,dfydP0, dfydP,dfydmu] = WSAQF(Y-xEq,P0,P,mu); dVydxEq= -dVydx;
    [Vx,dVxdx,dfxdP0, dfxdP,dfxdmu] = WSAQF(X-xEq,P0,P,mu); dVxdxEq= -dVxdx;
    dV =  Vy - Vx;  iinc =  dV > 0;
    f = sum(dV(iinc));
    dfdp = sum(reshape(dfydP0(iinc,:,:) - dfxdP0(iinc,:,:),sum(iinc),E^2)...
        * reshape(dP0dp,E^2,Np),1)...  % "dfdP0"
        + sum(reshape(dfydP(iinc,:,:) - dfxdP(iinc,:,:),sum(iinc),L*E^2)...
        * reshape(dPdp,L*E^2,Np),1)...   % "dfdP"
        + sum(reshape(dfydmu(iinc,:,:) - dfxdmu(iinc,:,:),sum(iinc),L*E) ...
        * reshape(dmudp,L*E,Np),1);   % "dfdmu"
    if nargin < 5
        dfdxEq = [];
    else
        dfdxEq = permute(dVydxEq-dVxdxEq,[2 1 3]);
        dfdxEq = dfdxEq(:,iNN);
        dfdxEq= sum(dfdxEq(:,iinc),2);
    end
    dfdx =  [ dfdxEq' dfdp];
    
end

end


function [c, ceq] = con(xEq,p,opt)
% Compute number of elements and reconstruct L
E = size(xEq,1);

[P0,P,~]  = p2WSAQF(p,E,opt.dWSAQF);
ceq = [];


if nargout <=2
    c = zeros((opt.dWSAQF+1)*E,2);
    
    
    % Formulating constraint
    c(1:E,1) = eig(P0) - opt.maxP;
    c(1:E,2) = -eig(P0) + opt.minP;
    
    for l=1:opt.dWSAQF
        c(l*E+1:(l+1)*E,1) = eig(P(:,:,l)) - opt.maxP;
        c(l*E+1:(l+1)*E,2) = -eig(P(:,:,l)) + opt.minP;
    end
    c=c(:);
    ceq = [];
else
    warning('no gradient implemented yet');
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

end


function [P0, P, mu, dP0dp, dPdp, dmudp] = p2WSAQF(p,D,L)
% Converts Parameter Vector p to P0, P and mu of a WSAQF
% In:
%    p     nP x 1
%    D     1 x 1
%    L     1 x 1
% Out:
%   P0      D x D
%   P       D x D x L
%   mu      D x L
%   dP0dp   D x D x nP
%   dPdp    D x D x L x nP
%   dmudp   D x L x nP
% D: Dimensionality of data
% Constants:
% nP = (L+1)*triD + L*D;
%{
clc; clear; close all; addpath(genpath('../mtools/'));
L = 3; D = 2; triD = D*(D+1)/2; nP = (L+1)*triD + L*D; p = rand(nP,1);
checkGrad(@p2WSAQF,1,1,4,{p,D,L});
checkGrad(@p2WSAQF,1,2,5,{p,D,L});
checkGrad(@p2WSAQF,1,3,6,{p,D,L});
%}
% Last modified: Jonas Umlauft, Armin Lederer, 01/2017


if ~isscalar(D)||~isscalar(L)
    error('wrong input dimensions');
end

triD = D*(D+1)/2; np = (L+1)*triD + L*D;
iP0 = 1:triD;
iP = triD+1:(L+1)*triD; iPi = reshape(iP,triD,L);
imu = (L+1)*triD+1:np; imui = reshape(imu,D,L);

if nargout <= 3
    P0 = Lvec2SPD(p(iP0));
    P = Lvec2SPD(p(iPi));
    mu = p(imui);
else
    dPdp = zeros(D,D,L,np);  dmudp = zeros(D,L,np); dP0dp = zeros(D,D,np);
    [P0, dP0dp(:,:,iP0)] = Lvec2SPD(p(iP0));
    [P, temp] = Lvec2SPD(p(iPi));
    dPdp(:,:,:,iPi) = reshape(temp,D,D,L,triD*L);
    mu = reshape(p(imui),D,L);
    dmudp(:,:,imu) = reshape(permute(eye(D*L),[1 3 2]),D,L,D*L);
end
end