clear; close all; 
load Testdata
L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);


% Define U and Usize , it will be used in both of the for loops
U = Undata;
Usize = size(U,1);


%Initiliaze an empty vector to add in the for loop
sum = zeros(n,n,n);

for j= 1:Usize
    u(:,:,:) = reshape(U(j,:),n,n,n);
    
    ut = fftn(u);
    % Append all the frequency data to one vector
    sum= sum+ut;
end


% Find the average and normalize 
uave = abs(fftshift(sum))/Usize;
uaven = uave/max(uave(:)); 
ind2 = find(uaven ==1);

% Find the central frequency in each direction by using
kx = Kx(ind2);ky = Ky(ind2);kz = Kz(ind2);

%Define a bandwidth, I played around with many values and found the highest
%optimized bandwith value
bandwith = 4.56;

%Gaussian Filter
filter = fftshift(exp(-1*bandwith * ((Kx-kx).^2 + (Ky-ky).^2 + (Kz-kz).^2)));
%empty 20x3 matrix to append the path of the marble
marble_path = zeros(20,3);

for times = 1:Usize    
    % Find x, y, z coordinates
    u(:,:,:)=reshape(U(times,:),n,n,n);
    unf = fftshift(fftshift(ifftn(fftn(u) .* filter)));
    [A, B] = max(unf(:));
    [X_path, Y_path, Z_path] = ind2sub([n,n,n], B); 
    %marble path appended
    marble_path(times, 1) = X(X_path, Y_path, Z_path);
    marble_path(times, 2) = Y(X_path, Y_path, Z_path);
    marble_path(times, 3) = Z(X_path, Y_path, Z_path);
  
end

plot3(marble_path(:,1),marble_path(:,2),marble_path(:,3));grid on

%where the marble is right now
last_stop = marble_path(20,:)`