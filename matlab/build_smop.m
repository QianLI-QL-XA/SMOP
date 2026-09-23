function build_smop()
%BUILD_BMOP  Compile the smop MEX interface.
%
%   build_smop
%
% Requires a C++17-capable MEX compiler (mex -setup c++), e.g. MinGW-w64
% or MSVC.  Eigen headers must be present at ../../third_party/eigen-3.4.0
% relative to the matlab/ folder (they are shipped with the package).

root = fileparts(mfilename('fullpath'));   % .../smop/matlab
inc  = fullfile(root, '..', 'include');
eig  = fullfile(root, '..', 'third_party', 'eigen-3.4.0');
if ~exist(eig, 'dir')
    eig = fullfile(root, '..', 'third_party', 'eigen');
end
if ~exist(fullfile(eig, 'Eigen', 'Core'), 'dir')
    error('smop:build', ...
        'Eigen headers not found at %s. Download Eigen 3.4.x into smop/third_party/.', eig);
end

mex('-O', ...
    ['-I', inc], ...
    ['-I', eig], ...
    fullfile(root, 'smop_mex.cpp'));
mex('-O', ...
    ['-I', inc], ...
    ['-I', eig], ...
    fullfile(root, 'smop_lasso_mex.cpp'));
fprintf('smop_mex / smop_lasso_mex built successfully.\n');
end
