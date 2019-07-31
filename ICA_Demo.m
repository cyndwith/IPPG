clc; clear all;close all;
% tRead = mmread('Dwith5.avi');
% mmwrite('faceVid.avi',tRead);
videoIn = VideoReader('faceVid.avi');
noFrames = videoIn.NumOfFrames;
figure;
frameCount = 0;
redMean = zeros(1,noFrames);
greenMean = zeros(1,noFrames);
blueMean = zero(1,noFrames);

x = frameCount;
while hasFrame(videoIn)
    frame = readFrame(videoIn); 
    frameCount = frameCount + 1
    if frameCount == 1
        x = frameCount;
    else
        x = [x frameCount];
    end
    
    red = frame(:,:,1);
    green = frame(:,:,2);
    blue = frame(:,:,3);
    
    % Show the R,G,B - Channels
    subplot(2,3,1);imshow(red);title('Red Channel');
    subplot(2,3,2);imshow(green);title('Green Channel');
    subplot(2,3,3);imshow(blue);title('Blue Channel');
    
    if frameCount == 1
        redMean = mean(mean(red));
        greenMean = mean(mean(green));
        blueMean = mean(mean(blue));
    else
        redMean = [ redMean mean(mean(red))];
        greenMean = [greenMean mean(mean(green))];
        blueMean = [blueMean mean(mean(blue))];
    end
    subplot(2,3,4);plot(x,redMean);title('Red Channel');
    subplot(2,3,5);plot(x,greenMean);title('Green Channel');
    subplot(2,3,6);plot(x,blueMean);title('Blue Channel');
   
    drawnow; 
end
% %%
%%%%%%%%%%%%%%%%%% Low pass filter %%%%%%%%%%%%%%%%%%%%%%
B = 1/30*ones(30,1);
redMean = filter(B,1,redMean);
greenMean = filter(B,1,greenMean);
blueMean = filter(B,1,blueMean);


%%%%%%%%%%%%%%%%% FFT of the input signals %%%%%%%%%%%%%%%
myFFT(redMean);
myFFT(greenMean);
myFFT(blueMean);

%%%%%%%%%% ICA Implementation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z = [redMean;greenMean;blueMean];
r = 3;
n = frameCount;

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


icaRED = Zica(1,:);
icaGREEN = Zica(2,:);
icaBLUE = Zica(3,:);

%icaRED = redMean;
%icaGREEN = greenMean;
%icaBLUE = blueMean;

% FFT of signal
icaSIG = icaGREEN;
L = size(icaGREEN,2);
icaSIG = icaSIG - mean(mean(icaSIG));
fftSIG = fft(icaSIG);
%L = size(fftSIG,2);
fftSIG = abs(fftSIG);
fftSIG2 = fftSIG(1:L/2+1);
fftSIG2(2:end-1) = 2*fftSIG2(2:end-1);

Fs = 29; % frames/second
freq = Fs*(0:(L/2))/L;
figure;plot(freq,fftSIG2);

% figure;
% subplot(2,1,1);plot(x,icaSIG);
% subplot(2,1,2);plot(x,abs(fftSIG));

figure;
plot(freq(1:1:50),abs(fftSIG2(1:50)));

% subplot(3,1,1);plot(x,fftYica(:,1));
% supplot(3,1,2);plot(x,fftYica(:,2));
% subplot(3,1,3);plot(x,fftYica(:,3));




