[y,Fs] = audioread('../../Data/metal.wav');
N = size(y,1); % Determine total number of samples in audio file


figure;
spectrogram(y, 1024, 3/4*1024, [], Fs, 'yaxis')
figure;
plot(psd(spectrum.periodogram,y,'Fs',Fs,'NFFT',length(y)));

% plot the signal waveform

N = length(fOut);                      % signal length
t = (0:N-1)/Fs;                     % time vector
figure;
plot(t, y, 'r')
xlim([0 max(t)])
ylim([-1.1*max(abs(y)) 1.1*max(abs(y))])
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Amplitude')
title('The signal in the time domain')
figure,
fy=fft(y);
L=length(y);
L2=round(L/2);
fa=abs(fy(1:L2)); % half is essential, rest is aliasing
fmax=Fs/2; % maximal frequency
fq=((0:L2-1)/L2)*fmax; % frequencies
plot(fq,fa);
