function [V, dVdx, dVdP] = SOS(x,P,dSOS)
% Computes the Weighted Sum of Asymmetric Quadratic Functions
% In:
%    P    Dm x Dm     Positive Symmetric matrix
%    x    E x N       Point where function is evaluated
%    dSOS 1 x 1       Degree of SOS
% OR exMat Dm x E     Combinations of exponents matrix
% Out:
%    V    N x 1       Function value
%    dVdx N x E x N       Derviative w.r.t x
%    dVdP N x Dm x Dm
% E: Dimensionality of x
% Dm: Dimensionality of monomial
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

[E,N] = size(x); Dm = size(P,1);

if isscalar(dSOS)
    exMat = getExpoMatrix(E,dSOS);
else
    exMat = dSOS;
end

if size(exMat,1) ~=Dm || size(P,2) ~= Dm ||size(exMat,2) ~=E
    error('wrong input dimensions');
end

if nargout == 1
    mon = getMonomial(x,exMat);
else
    [mon, dmondx]= getMonomial(x,exMat);
end


V = sum(permute(sum(permute(P,[3 1 2]).*mon',2),[1 3 2]).*mon',2);

if nargout > 1
    dVdx = zeros(E,N,N); iNN = 1:N+1:N^2;
    for e = 1:E
        dmondx_temp = permute(dmondx(:,:,e,:),[1 2 4 3]); %Dm x N x N
        dVdx(e,iNN) = 2*sum(permute(sum(permute(P,[3 1 2]).*dmondx_temp(:,iNN)',2),[1 3 2]).*mon',2);
    end
    dVdx = permute(dVdx,[2 1 3]);
end

if nargout > 2
    dVdP = zeros(N,Dm,Dm);
    for dm1=1:Dm
        for dm2=1:Dm
            dVdP(:,dm1,dm2) = dVdP(:,dm1,dm2) + (mon(dm1,:).*mon(dm2,:))';
        end
    end
end
end
 
% if nargout > 1
%     dVdx = zeros(E,N,N); dmondx_perm = permute(dmondx,[1 3 2 4]);
%     for dm1=1:Dm
%         for dm2=1:Dm
%             for e = 1:E
%                 dVdx(e,iNN) = dVdx(e,iNN) + 2*permute(dmondx_perm(dm2,e,iNN),[1 3 2]).*P(dm1,dm2).*mon(dm1,:);
%             end
%         end
%     end
%     dVdx = permute(dVdx,[2 1 3]);
% end

%     if nargout > 1,dVdx = zeros(N,E,N);end
%     if nargout > 1,dVdP = zeros(N,E,E);end
%     for n = 1:N
%         if nargout > 1
%             dVdx(n,:,n) = 2*permute(dmondx(:,n,:,n),[3 1 2 4])*P*mon(:,n);
%             if nargout > 2
%                 dVdP(n,:,:) = mon(:,n)*mon(:,n)';
%             end
%         end
%     end    
 
function [mon, dmondx,dmondxdx]= getMonomial(x,exMat)
%%GETMONOMIAL computes all monomials for x
% In:
%   x     E  x N         input
%   ExMat Dm x E         Matrix of all combinations
% Out
%  mon      Dm x N       all monomials
%  dmondx   Dm x N x E x N      Derivative w.r.t x
%  dmondxdx Dm x N x E x N x E x N  Derivative w.r.t x (twice)
% E: Dimensionality of x
% Dm: Dimensionality of monomial
%{
clear all, close all, rng default; addpath('./mtools');
N = 3; E = 2; dSOS = 2; x = rand(E,N); exMat = getExpoMatrix(E,dSOS);
[m2, dmdx,dmdxdx]= getMonomial(x,exMat);
[n,num,ana] = checkGrad(@getMonomial,1,1,2,{x,exMat});
[n,num,ana] = checkGrad(@getMonomial,1,2,3,{x,exMat});
%}
% Last modified: Jonas Umlauft, 05/2017

[E,N] = size(x);
x = x+10e-20;

Dm = size(exMat,1);

if size(exMat,2) ~=E
    error('wrong input dimensions');
end
exMatp = permute(exMat,[1 3 2]);
xp = permute(x,[3 2 1]);
monall = xp.^exMatp; % Dm x N x E
mon = prod(monall,3);

% Compute derivatives if necessary
if nargout > 1
    deriv_new = exMatp.*xp.^(exMatp-1);  % Dm x N x E
    
    iNN = 1:N+1:N^2; dmondx = zeros(Dm,E,N,N);
    
    dmondx(:,:,iNN) = permute(deriv_new.*mon./monall,[1 3 2]);
    dmondx = permute(dmondx,[1 3 2 4]);
end

% % Compute derivatives if necessary
if nargout > 2
    dmondxdx = zeros(Dm,E,E,N,N,N); iNNN =1:N^2+N+1:N^3;
    iEE = 1:E+1:E^2; niEE = setdiff(1:E^2,iEE);
    deriv2 = zeros(Dm,N,E,E);
    deriv2(:,:,iEE) =  exMatp.*(exMatp-1).*xp.^(exMatp-2);
    if E ~= 1,deriv2(:,:,niEE) = exMatp.*xp.^(exMatp-1);end
    
    %     [iE,jE] =  ind2sub(E,iEE); [niE,njE] =  ind2sub(E,niEE);
    %     dmondxdx_new(:,iE,jE,iNNN) = permute(deriv2_new(:,:,iEE).*(mon./monall(:,:,iE)),[1 3 4 2]);
    %     if E ~= 1
    %     dmondxdx_new(:,niE,njE,iNNN) = permute(deriv2_new(:,:,niE,njE).*deriv2_new(:,:,njE,niE).*(mon./monall(:,:,niE)./monall(:,:,njE)),[1 3 4 2]);
    %     end
    for e1=1:E
        for e2=1:E
            if e1 == e2
                dmondxdx(:,e1,e2,iNNN)=permute(deriv2(:,:,e1,e2).*...
                    mon./monall(:,:,e1),[1 3 4 2]);
            else
                dmondxdx(:,e1,e2,iNNN)=permute(deriv2(:,:,e2,e1).*...
                    deriv2(:,:,e1,e2).*mon./monall(:,:,e1)./monall(:,:,e2),[1 3 4 2]);
            end
        end
    end
    
    dmondxdx = permute(dmondxdx,[1 4 2 5 3 6]);
    
end

end

function comb=getExpoMatrix(E,D)
% Computes all possible combinations to choose E integers (not distinct)
% such that they add up to a number less or equal to D
% In: 
%   E    1 x 1 
%   D    1 x 1 
% Out: 
%  Comb  D+1 x ? 
% Last edited: 09/2016, Jonas Umlauft, Armin Lederer
comb = [];
for d = 0:D
    c = nchoosek(1:d+E-1,E-1);
    m = size(c,1);
    t = ones(m,d+E-1);
    t(repmat((1:m).',1,E-1)+(c-1)*m) = 0;
    u = [zeros(1,m);t.';zeros(1,m)];
    v = cumsum(u,1);
    comb = [comb; diff(reshape(v(u==0),E+1,m),1).'];
end
comb = comb(2:end,:);
end