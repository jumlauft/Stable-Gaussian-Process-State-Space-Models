function savej(filenameOFsavej,name_vars,vars)
%SAVEJ Saves variables to file (needed within parfor)
%In: 
%   fname      char str     file name to which is saved
%   name_vars  {N x 1}      Names of variable
%   vars       {N x 1}      variables
% Last modified: Jonas Umlauft 2017-05

if numel(name_vars) ~=numel(vars)
    error('wrong input dimension');
end
N = numel(name_vars); 

for n=1:N
    eval([name_vars{n},'= vars{n};']);
end
clear vars name_vars

save(filenameOFsavej);

end

