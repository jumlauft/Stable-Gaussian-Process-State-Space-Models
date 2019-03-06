function A = errArea(Xsim,Xdemo,uplim)
% Computes the error area between Ntraj trajectories in 2D
% In:
%   Xsim   {Ntraj} 2 x ?    
%   Xdemo   {Ntraj} 2 x ?
%  uplim     1 x 1         Upper Limit (default 1e4)
% Out: 
%  A       Ntraj           Error Area for each trajectory
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019
if ~exist('uplim','var'), uplim = 1e3; end

 Ntraj = length(Xsim);
 if Ntraj~=length(Xdemo)
     warning('wrong input dimensions');
 end
% Fill default value

A = zeros(Ntraj,1);
 for n=1:Ntraj
        min_x = min(min(min(Xsim{n}(1,:)),uplim),min(Xdemo{n}(1,:)));
        min_y = min(min(min(Xsim{n}(2,:)),uplim),min(Xdemo{n}(2,:)));

        Xsim{n} = [Xsim{n},zeros(2,1)];
       
        x=[Xdemo{n}(1,:),flip(Xsim{n}(1,:))]';
        y=[Xdemo{n}(2,:),flip(Xsim{n}(2,:))]';
        
        x=(x-min_x); 
        y=(y-min_y);
        xy_max = max(max(x),max(y));
        x=x/xy_max*uplim;
        y=y/xy_max*uplim;
        Bw=poly2mask(x,y,uplim,uplim);
        A(n)=sum(sum(Bw))*(xy_max/uplim)^2;
 end
 
