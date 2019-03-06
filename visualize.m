% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

%% Setup Grid
Ndte = floor(nthroot(Nte,E)); 
Xte = ndgridj(min(Xtr,[],2)-gmte, max(Xtr,[],2)+gmte,Ndte*ones(E,1)) ;
Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte);
Ndemo = length(Xtraj_unst);

%% GPSSM without stabilization
figure; hold on; axis tight;
title('GPSSM without stabilization')
surf(Xte1,Xte2,reshape(sqrt(sum(varGPte.^2,1)),Ndte,Ndte)-1e4,'EdgeColor','none','FaceColor','interp');colormap(flipud(parula));
streamslice(Xte1,Xte2,reshape(mGPte(1,:),Ndte,Ndte)-Xte1,reshape(mGPte(2,:),Ndte,Ndte)-Xte2,dss); %set(h, 'Color', 'r' );
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'color','black')

%% GPSSM with stabilization
figure; hold on; axis tight
title('Stabilized GPSSM + Trajectories')
contour(Xte1,Xte2,reshape(Vclf(Xte),Ndte,Ndte),'g','LevelList',...
    Vclf([linspace(min(Xtr(1,:))-gmte, max(Xtr(1,:))+gmte,dscont); zeros(1,dscont)]));
streamslice(Xte1,Xte2,reshape(mGPstabte(1,:),Ndte,Ndte)-Xte1,reshape(mGPstabte(2,:),Ndte,Ndte)-Xte2,dss);%set(h, 'Color', 'r' );
for ndemo=1:length(Xtraj), plot(Xtraj{ndemo}(1,:),Xtraj{ndemo}(2,:),'r'); end
for ndemo=1:length(Xtraj_unst), plot(Xtraj_unst{ndemo}(1,:),Xtraj_unst{ndemo}(2,:),'m');
    text(Xtraj_unst{ndemo}(1,1),Xtraj_unst{ndemo}(2,1),num2str(ndemo));
end
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'k','AutoScale','off');

