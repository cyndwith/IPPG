clc;clear all;close all;

x = [0:0.01:10];
y = sin(2*3.14*x);
n = size(x,2);
y1 = sin(2*3.14*x) + rand(1,n)*2;
y2 = sin(5*3.14*x) + rand(1,n)*2;
y3 = sin(7*3.14*x) + rand(1,n)*2;

inY1 = y1*0.7 + y2*0.3 + y3*0.4 + rand(1,n)*2;
inY2 = y1*0.2 + y2*0.6 + y3*0.1 + rand(1,n)*2;
inY3 = y1*0.8 + y2*0.2 + y3*0.4 + rand(1,n)*2;

% Apply ICA to y1,y2,y3
Z = [inY1;inY2;inY3];
r = 3;

% Perform ICA
[Zica A T mu] = myICA(Z,r);
Zr = T \ pinv(A) * Zica + repmat(mu,1,n);

% Plot indpendent components
figure;
for i = 1:r
    subplot(r,1,i);
    plot(Zica(i,:),'b');
    grid on;
    ylabel(sprintf('Zica(%i,:)',i));
end



% FFT of signal

icaGREEN = Zica(3,:);
%icaGREEN = inY1;
icaSIG = icaGREEN;
L = size(icaGREEN,2);
icaSIG = icaSIG - mean(mean(icaSIG));
fftSIG = fft(icaSIG);
%L = size(fftSIG,2);
fftSIG = abs(fftSIG);
fftSIG2 = fftSIG(1:L/2+1);
fftSIG2(2:end-1) = 2*fftSIG2(2:end-1);

Fs = 100; % frames/second
freq = Fs*(0:(L/2))/L;
figure;plot(freq,fftSIG2);
title('FFT of Signal (One Sided)');

figure;title('Zoomed in FFT');
plot(freq(1:1:50),abs(fftSIG2(1:50)));

figure;title('Input Signal');
subplot(3,1,1);plot(x,inY1);
subplot(3,1,2);plot(x,inY2);
subplot(3,1,3);plot(x,inY3);


