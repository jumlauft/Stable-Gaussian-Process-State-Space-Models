function [V, dVdx,dVdP0, dVdP,dVdmu] = WSAQF(x,P0,P,mu)
% Computes the Weighted Sum of Asymmetric Quadratic Functions (WSAQF)
% In:
%   x      E x N         Points where function is evaluated
%   P0     E x E         Symmetric square matrix
%   P      E x E x L     Asymmetric square matrices
%   mu     E x L         Asymmetric shift
% Out:
%   V       N x 1         Function value
%   dVdx    N x E x N         Derviative w.r.t x
%   dVdP0   N x E x E         Derviative w.r.t P0
%   dVdP    N x E x E x L     Derviative w.r.t P
%   dVdmu   N x E x L         Derviative w.r.t mu
% E: Dimensionality of data
% L: Number of asymmetric centers
% For details regarinding WSAQF refer to
%   Khansari-Zadeh, S. M. & Billard, A.
%   Learning control Lyapunov function to ensure stability of dynamical
%   system-based robot reaching motions Robotics and Autonomous Systems,
%   Elsevier, 2014, 62, 752-765
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

% Verfiy Sizes
[E,L] = size(mu);
N = size(x,2);

if size(P0,1) ~= E || size(P0,2) ~= E ||size(P,1) ~= E ||...
        size(P,2) ~= E || size(P,3) ~= L ||  size(x,1) ~= E
    error('wrong input dimensions');
end

% Compute function value (batch)
Vl = zeros(L,N);
for l = 1:L
    Vl(l,:) = sum(permute(sum(permute(P(:,:,l),[3 1 2]).*x',2),[1 3 2]).*(x-mu(:,l))',2);
end
V0 = sum(permute(sum(permute(P0,[3 1 2]).*x',2),[1 3 2]).*x',2);
iL = Vl > 0;
V = V0 + sum((Vl.*iL).^2,1)';


% Compute Derivatives w.r.t. input if needed
if nargout > 1
    dVdx = zeros(E,N,N); iNN = 1:N+1:N^2;
    dVdx(:,iNN) = 2*P0*x;
    for l = 1:L
        dVldx = 2*P(:,:,l)*x - P(:,:,l)*mu(:,l);
        dVdx(:,iNN) = dVdx(:,iNN) + 2*dVldx.*(Vl(l,:).*iL(l,:));
    end
    dVdx = permute(dVdx,[2 1 3]);
end
        

% Compute Derivatives w.r.t. Parameters if needed
if nargout > 2
    dVdP0 = zeros(N,E,E); dVdP = zeros(N,E,E,L); dVdmu = zeros(N,E,L);
    for n = 1:N
    dVdP0(n,:,:) = permute(x(:,n)*x(:,n)',[3 1 2]);
    for l = 1:L
        if Vl(l,n) > 0
            dVldmu = -P(:,:,l)*x(:,n);
            dVdmu(n,:,l) = 2*Vl(l,n)*dVldmu;
            
            dVldP = x(:,n)*(x(:,n)-mu(:,l))';
            dVdP(n,:,:,l) = 2*Vl(l,n) * dVldP;
        end
    end
    end
end

% % Compute Derivatives w.r.t. Parameters and input if needed
% if nargout > 5
%     dVdxdP0 = zeros(E,E,E); dVdxdP = zeros(E,E,E,L); dVdxdmu = zeros(E,E,L);
%     for l = 1:L
%         if Vl(l) >0
%             xPmPmu = (2*x'*P(:,:,l) - (P(:,:,l)*mu(:,l))')';
%             
%             for e1=1:E
%                 dVdxdmu(:,e1,l) = -P(:,e1,l) * 2*Vl(l)  -2*x'*P(:,e1,l) * xPmPmu;
%                 if l==1,  dVdxdP0(e1,e1,:) = 2*permute(x,[2 3 1]); end
%                 for e2=1:E
%                     dVdxdP(:,e1,e2,l) =  2*x(e1)*(x(e2)-mu(e2,l))  *  xPmPmu;
%                     dVdxdP(e2,e1,e2,l) = dVdxdP(e2,e1,e2,l) + 4*Vl(l)*x(e1);
%                     dVdxdP(e1,e1,e2,l) = dVdxdP(e1,e1,e2,l) - 2*Vl(l)*mu(e2,l);
%                 end
%             end
%         end
%     end
% end

end
