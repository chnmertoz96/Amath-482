
% Part 1
clear all; close all; clc;

load handel
v = y'/2;
v = v(1:length(v)-1);

% Fourier Transform
L = 9; n = length(v);
t2 = linspace(0, L, n+1); t= t2(1:n);
k = (2*pi/L) * [0:n/2-1 -n/2:-1]; ks = fftshift(k);   

tslide= 0:0.1:9;
spectrum1 = [];
% Gaussian Window, a = 1
for i=1:length(tslide)
    g = exp(-1*(t-tslide(i)).^2);
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum1 = [spectrum1; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum1.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Gaussian Window, a = 1");


spectrum2 = [];
%Gaussian Window, a = 10
for i=1:length(tslide)
    g = exp(-10*(t-tslide(i)).^2);
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum2 = [spectrum2; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum2.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Gaussian Window, a = 10")

spectrum3 = [];
%Gaussian Window, a = 1000
for i=1:length(tslide)
    g = exp(-1000*(t-tslide(i)).^2);
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum3 = [spectrum3; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum3.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Gaussian Window, a = 1000")

spectrum4 = [];
om = 0.1;
%Mexican Hat Wavelet, omega = 0.1
for i=1:length(tslide)
    g = (2/ (sqrt(3*om) * pi^(0.25))).* (1-((t-tslide(i))/om).^2) .* exp(-((t-tslide(i)).^2)/(2 * om^2));
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum4 = [spectrum4; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum4.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Mexican Hat, omega = 0.1")


spectrum5 = [];
om = 0.01;
%Mexican Hat Wavelet, omega = 0.01
for i=1:length(tslide)
    g = (2/ (sqrt(3*om) * pi^(0.25))).* (1-((t-tslide(i))/om).^2) .* exp(-((t-tslide(i)).^2)/(2 * om^2));
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum5 = [spectrum5; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum5.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Mexican Hat, omega = 0.01")

spectrum6 = [];
sg = 0.1; 
%Shannon Wavelett, sigma = 0.1
for i=1:length(tslide)
    g = abs(t-tslide(i)) < sg;
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum6 = [spectrum6; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum6.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Shannon Wavelet, Sigma = 0.1")

spectrum7 = [];
sg = 0.01; 
%Shannon Wavelett, sigma = 0.01
for i=1:length(tslide)
    g = abs(t-tslide(i)) < sg;
    filter_g = g.*v;
    fourier_vg = fft(filter_g);  
    spectrum7 = [spectrum7; abs(fftshift(fourier_vg))];
end
figure;
pcolor(tslide, ks./(2*pi),spectrum7.'), shading interp, colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Shannon Wavelet, Sigma = 0.01")

% Part 2.1
close all; clear all; clc;

[y,Fs] = audioread('music1.wav');
tr_piano=length(y)/Fs; % record time in seconds
%plot((1:length(y))/Fs,y);
%xlabel('Time [sec]'); ylabel('Amplitude');
%title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);
%figure(2)

% Fourier Transform
n = length(y);
k = (2*pi/tr_piano)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);
t = (1:n)/Fs;
tslide = 0:0.2:tr_piano;

spectrum = [];
score = [];

% Gaussian Window and Score
for i=1:length(tslide)
    g = exp(-100*(t-tslide(i)).^2);
    filter_g = g.*y';
    fourier_vg = fft(filter_g);  
    spectrum = [spectrum; abs(fftshift(fourier_vg))];
    [M,I] = max(abs((fourier_vg)));
    score = [score; abs(k(I))/(2*pi)];
    
end

figure()
pcolor(tslide,(ks/(2*pi)),spectrum.'), shading interp
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Piano");
ylim([100 500 ])

% Part 2.2
close all; clear all; clc;

[y,Fs] = audioread('music2.wav');
tr_rec=length(y)/Fs; % record time in seconds
%plot((1:length(y))/Fs,y);
%xlabel('Time [sec]'); ylabel('Amplitude');
%title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);
%figure(2)
% Fourier Transform
n = length(y);
k = (2*pi/tr_rec)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);
t = (1:n)/Fs;
tslide = 0:0.2:tr_rec;

spectrum = [];
score = [];

% Gaussian Window and Scor
for i=1:length(tslide)
    g = exp(-100*(t-tslide(i)).^2);
    filter_g = g.*y';
    fourier_vg = fft(filter_g);  
    spectrum = [spectrum; abs(fftshift(fourier_vg))];
    [M,I] = max(abs((fourier_vg)));
    score = [score; abs(k(I))/(2*pi)];
    
end

figure()
pcolor(tslide,(ks/(2*pi)),spectrum.'), shading interp,colormap(hot)
xlabel("Time (s)");ylabel("Frequency (Hz)"); title("Piano");
ylim([600 1300])