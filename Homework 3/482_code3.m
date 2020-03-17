% Case 1
clear all; close all ;clc;

%load the images
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
% components of the video frame
[x,y,color,l] = size(vidFrames1_1);
[compx1,compy1,compx2,compy2,compx3,compy3] = deal(zeros(1,l));
filter = 0.95;



for k = 1:l
        X = double(rgb2gray(vidFrames1_1(:,:,:,k)));
        % frames
        X(1:220,:) = 0;
        X(:, 1:320) = 0;
        X(:, 390:end) = 0;
        Y = double(rgb2gray(vidFrames2_1(:,:,:,k)));
        Y(1:120,:) = 0;
        Y(:, 1:260) = 0;
        Y(:, 350:end) = 0;
        Z = double(rgb2gray(vidFrames3_1(:,:,:,k)));
        Z(1:270,:) = 0;
        Z(:, 1:290) = 0;
        Z(:, 440:end) = 0;
        
        
        % the components
        M = max(X(:));
        [maxX1,maxX2] = find(X >= M * filter);
        compx1(k) = mean(maxX1);
        compy1(k) = mean(maxX2); 
        M = max(Y(:));
        [maxY1,maxY2] = find(Y >= M * filter);
        compx2(k) = mean(maxY1);
        compy2(k) = mean(maxY2); 
        M = max(Z(:));
        [maxZ1,maxZ2] = find(Z >= M * filter);
        compx3(k) = mean(maxZ1);
        compy3(k) = mean(maxZ2); 
        
        
end


% Adding the components to a new matrix
M = [compx1; compy1; compx2; compy2; compx3; compy3];

% singular value decomposition
[a,b] = size(M); mean = mean(M,2); M =  M - repmat(mean,1,b);

[u,s,v] = svd(M,'econ');


subplot(1,2,1)
plot(v*s)
xlabel('Time');ylabel('Location')
legend('1','2','3','4','5','6')
subplot(1,2,2)
plot(diag(s)/sum(diag(s)))
xlabel('PC');ylabel('Energy Level')


% Case 2
clear all; close all ;clc;

%load the images
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')
% components of the video frame
[x,y,color,l] = size(vidFrames1_2);
[compx1,compy1,compx2,compy2,compx3,compy3] = deal(zeros(1,l));
filter = 0.9;

for k = 1:l
        X = double(rgb2gray(vidFrames1_2(:,:,:,k)));
        % frames
        X(1:200,:) = 0;
        X(:, 1:300) = 0;
        X(:, 401:end) = 0;
        X(:,400:end) = 0;
        Y = double(rgb2gray(vidFrames2_2(:,:,:,k)));
        Y(1:50,:) = 0;
        Y(:, 1:200) = 0;
        Y(370:end,:) = 0;
        Y(:, 401:end) = 0;
        Z = double(rgb2gray(vidFrames3_2(:,:,:,k)));
        Z(1:200,:) = 0;
        Z(320:end,:) = 0;
        Z(:, 1:280) = 0;
        Z(:, 501:end) = 0;
        
        
        % the components
        M = max(X(:));
        [maxX1,maxX2] = find(X >= M * filter);
        compx1(k) = mean(maxX1);
        compy1(k) = mean(maxX2); 
        M = max(Y(:));
        [maxY1,maxY2] = find(Y >= M * filter);
        compx2(k) = mean(maxY1);
        compy2(k) = mean(maxY2); 
        M = max(Z(:));
        [maxZ1,maxZ2] = find(Z >= M * filter);
        compx3(k) = mean(maxZ1);
        compy3(k) = mean(maxZ2); 
        
end

% Adding the components to a new matrix
M = [compx1; compy1; compx2; compy2; compx3; compy3];

% singular value decomposition
[a,b] = size(M); mean = mean(M,2); M =  M - repmat(mean,1,b);

[u,s,v] = svd(M,'econ');

subplot(1,2,1)
plot(v*s)
xlabel('Time');ylabel('Location')
legend('1','2','3','4','5','6')
subplot(1,2,2)
plot(diag(s)/sum(diag(s)))
xlabel('PC');ylabel('Energy Level')


% Case 3
clear all; close all ;clc;

%load the images
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
% components of the video frame
[x,y,color,l] = size(vidFrames1_3);
[compx1,compy1,compx2,compy2,compx3,compy3] = deal(zeros(1,l));
filter = 0.98;
l = l-10;
for k = 1:l
        X = double(rgb2gray(vidFrames1_3(:,:,:, k)));
         % frames
        X(1:200,:) = 0;
        X(:, 1:250) = 0;
        X(:, 401:end) = 0;
        Y = double(rgb2gray(vidFrames2_3(:,:,:, k)));
        Y(1:200,:) = 0;
        Y(:, 1:200) = 0;
        Y(:, 401:end) = 0;
        Z = double(rgb2gray(vidFrames3_3(:,:,:,k)));
        Z(1:200,:) = 0;
        Z(:, 1:280) = 0;
        Z(:, 501:end) = 0;
        
        
        % the components
        M = max(X(:));
        [maxX1,maxX2] = find(X >= M * filter);
        compx1(k) = mean(maxX1);
        compy1(k) = mean(maxX2); 
        M = max(Y(:));
        [maxY1,maxY2] = find(Y >= M * filter);
        compx2(k) = mean(maxY1);
        compy2(k) = mean(maxY2); 
        M = max(Z(:));
        [maxZ1,maxZ2] = find(Z >= M * filter);
        compx3(k) = mean(maxZ1);
        compy3(k) = mean(maxZ2); 
        
end

% Adding the components to a new matrix
M = [compx1; compy1; compx2; compy2; compx3; compy3];

% singular value decomposition
[a,b] = size(M); mean = mean(M,2); M =  M - repmat(mean,1,b);

[u,s,v] = svd(M,'econ');

subplot(1,2,1)
plot(v*s)
xlabel('Time');ylabel('Location')
legend('1','2','3','4','5','6')
xlim([0 205])
subplot(1,2,2)
plot(diag(s)/sum(diag(s)))
xlabel('PC');ylabel('Energy Level')

% Case 4
clear all; close all ;clc;

%load the images
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')
% components of the video frame
[x,y,color,l] = size(vidFrames1_4);
[compx1,compy1,compx2,compy2,compx3,compy3] = deal(zeros(1,l));
filter = 0.9;

for k = 1:l
        X = double(rgb2gray(vidFrames1_4(:,:,:,k)));
        % frames
        X(1:200,:) = 0;
        X(:, 1:300) = 0;
        X(:, 501:end) = 0;
        Y = double(rgb2gray(vidFrames2_4(:,:,:,k)));
        Y(1:50,:) = 0;
        Y(:, 1:220) = 0;
        Y(:, 411:end) = 0;
        Z = double(rgb2gray(vidFrames3_4(:,:,:,k)));
        Z(1:150,:) = 0;
        Z(:, 1:300) = 0;
        Z(:, 511:end) = 0;
        
        
        
        M = max(X(:));
        [maxX1,maxX2] = find(X >= M * filter);
        compx1(k) = mean(maxX1);
        compy1(k) = mean(maxX2); 
        M = max(Y(:));
        [maxY1,maxY2] = find(Y >= M * filter);
        compx2(k) = mean(maxY1);
        compy2(k) = mean(maxY2); 
        M = max(Z(:));
        [maxZ1,maxZ2] = find(Z >= M * filter);
        compx3(k) = mean(maxZ1);
        compy3(k) = mean(maxZ2); 
        
end

% Adding the components to a new matrix
M = [compx1; compy1; compx2; compy2; compx3; compy3];

% singular value decomposition
[a,b] = size(M); mean = mean(M,2); M =  M - repmat(mean,1,b);

[u,s,v] = svd(M,'econ');


subplot(1,2,1)
plot(v*s)
xlabel('Time');ylabel('Location')
legend('1','2','3','4','5','6')
subplot(1,2,2)
plot(diag(s)/sum(diag(s)))
xlabel('PC');ylabel('Energy Level')