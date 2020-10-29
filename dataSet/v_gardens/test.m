%% Example Title
% Summary of example objective

% h = cpselect(imSrc,imDst); 
% h = cpselect(imSrc,imDst,movingPoints, fixedPoints); 

clear all;
close all;

%load SrcPts.mat;
%load DstPts.mat;

load 'C:\WINDOWS.old\Users\AGA-PC\Documents\GitHubRepos\hpatches-sequences-release.tar\hpatches-sequences-release\v_dogman\keyPoints.mat';

SrcPts = zeros(length(kpts),2);

for i = 1 : length(kpts)
    SrcPts(i,1) = kpts(i,2);
    SrcPts(i,2) = kpts(i,3);
end


imSrc = imread('C:\WINDOWS.old\Users\AGA-PC\Documents\GitHubRepos\SuperGlue\dogman\1.png');
imDst = imread('C:\WINDOWS.old\Users\AGA-PC\Documents\GitHubRepos\SuperGlue\dogman\2.png');

% h = cpselect(imSrc,imDst); 
% h = cpselect(imSrc,imDst,movingPoints, fixedPoints);
%save SrcPts.mat SrcPts
%save DstPts.mat DstPts

%gardens
%H1_2 = [ 2.2787 0.023843 -30.321
%0.58793 1.9158 -459.28
%0.0012782 -6.6868e-06 0.99971 ];

%dogman
H1_2 = [0.49838 -0.015725 33.278
        -0.18045 0.77392 59.799
        -0.00064863 -4.2793e-05 0.99978];


hom_Src_pts = cart2hom(SrcPts).';

ret = H1_2 * hom_Src_pts;
ret2 = hom2cart(ret.');

h = cpselect(imSrc,imDst,SrcPts, ret2); 

for i = 1 : length(kpts)
   kpts(i,8) = ret2(i,1);
   kpts(i,9) = ret2(i,2);
end
