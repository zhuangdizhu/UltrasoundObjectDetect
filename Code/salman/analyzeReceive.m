clearvars;
fontsize = 18;
close all; 
% [y,Fs, bits] = wavread('receive.wav');  
% wavread is replaced by audioread() for MATLAB 2017. --Judy
%[y,Fs] = audioread('../Data/metal.wav'); %
[y,fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/wood/2018-Feb-21_11-19-13.wav');
% pOrig = audioplayer(f,fs);
% pOrig.play; 
figure
plot(psd(spectrum.periodogram,y,'Fs',fs,'NFFT',length(y)));
ylim([-250 0])
set(gca,'FontSize', fontsize)

n = 7;
beginFreq = 17000 / (fs/2);
endFreq = 19000 / (fs/2);
[b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); %btype, analog -- Judy
y = filter(b, a, y);
N = length(y);                  % signal length



y=y'; 
fy=fft(y);

L=length(y);
L2=round(L/2);
fa=abs(fy(1:L2)); % half is essential, rest is aliasing

fmax=fs/2; % maximal frequency
fq=((0:L2-1)/L2)*fmax; % frequencies
size(fq)
 
figure
plot(psd(spectrum.periodogram,y,'Fs',fs,'NFFT',length(y))); 
ylim([-250 0])
set(gca,'FontSize', fontsize)
 %{
  plot(fq,fa);
  temp = 1;
  y_f = zeros(1,2000);
   for j=1:1:size(fq)
      if (fq(1,j)>= 17000) && (fq(1,j)<=19000)
         y_f(1,temp) =    fa(1,j);
         temp = temp+1;
      end
   end
   size(y_f)
   plot(fa)
      y_f(1,j) = fq(1,j+17000);
  end
  plot(y_f)
  size(y_f)
 % energy & spectral, MFCC, 
 %}
 

% N=length(y)
% % N = number of samples
% 
% 
% f=Fs/N.*(0:N-1);
% % calculate each frequency component
% 
% Y=fft(y,N);
% Y=abs(Y(1:N))./(N/2);
% plot(f,Y)
