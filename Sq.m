function [V, dVdx, dVdP,dVdxdP] = Sq(x,P)
% SIMPLESQ: Computes the Squared function V = x'Px and derivatives
% In:
%    x      E x N       Points where function is evaluated
%    P      E x E       Symmetric square matrix
% Out:
%    V      N x 1       Function value
%    dVdx   N x E x N   Derviative w.r.t x
%    dVdP   N x E x E   Derviative w.r.t P
%    dVdxdP N x E x E   Derviative w.r.t x and P
% E: Dimensionality of x
% N: Number of data points
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

% Check inputs
[E,N] = size(x);
if size(P,1) ~= E || size(P,2) ~= E
    error('wrong input dimensions');
end
V = sum(permute(sum(permute(P,[3 1 2]).*x',2),[1 3 2]).*x',2);
if nargout > 1
    dVdx = zeros(E,N,N); iNN = 1:N+1:N^2;
    dVdx(:,iNN) = 2*P*x;
    dVdx = permute(dVdx,[2 1 3]);
end
if nargout > 2,dVdP = zeros(N,E,E); end
if nargout > 3,dVdxdP = zeros(E,E,E,N); end
for n = 1:N
    if nargout > 2, dVdP(n,:,:) = permute(x(:,n)*x(:,n)',[3 1 2]); end
    if nargout > 3
        for e=1:E
            dVdxdP(:,e,e,n) = dVdxdP(:,e,e,n) + x(:,n);
            dVdxdP(e,:,e,n) = dVdxdP(e,:,e,n) + x(:,n)';
        end
    end
end



