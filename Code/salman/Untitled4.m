%% Plot both audio channels
% clearvars;
 close all;
% clear all;
%cd /Users/judy/Projects/UltraSoundObjectDetection/Code/

PLOT1 = 1;
PLOT2 = 0;
PLOT3 = 0;
PLOT4 = 0; 
[y,Fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/onbox/zhuangdi/plastic_box_0318/plastic_box100.wav'); 
%[y,Fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/tone1.wav'); 
 
N = length(y); % Determine total number of samples in audio file
 
%{
figure;
stem((1:N), y);
title(' Channel');
plot spectrum
%}
  
%% Design a bandpass filter that filters out between 700 to 12000 Hz
n = 7;
beginFreq = 17000 / (Fs/2);
endFreq = 19000 / (Fs/2);
[b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); %btype, analog -- Judy
fOut = filter(b, a, y);
 

if PLOT1 == 1  
    
    %{
    figure;
    subplot(2,1,1)
    plot(y)
    ylim([-00.03 0.03])
    subplot(2,1,2)
    plot(fOut)
    ylim([-00.03 0.03])
    %}
    %spectrogram(fOut, 1024, 3/4*1024, [], Fs, 'yaxis')
    %plot(psd(spectrum.periodogram,fOut,'Fs',Fs,'NFFT',length(fOut)));
    
    figure
    subplot(2,1,1)
    periodogram(fOut);
    subplot(2,1,2)
    %periodogram(fOut);
    pwelch(fOut)
    
    figure
    subplot(2,1,1)
    N = length(fOut);
    window = hanning(N, 'periodic'); 
    periodogram(fOut, window, N, Fs, 'power');
    subplot(2,1,2)
    pwelch(fOut, 1024, 512, [], Fs);
    
    %%{
    N = length(pxx);
    plot(1:N,10*log10(pxx)) 
    use_range = floor(N*0.65) : floor(N * 0.9);
    p = pxx(use_range);
    
    figure
    plot(10*log10(p))
    %}
end
return
N = length(fOut);                       % signal length
% plot the signal waveform
if PLOT2 == 1
    t = (0:N-1)/Fs;                         % time vector
    %%{
    figure;
    plot(t, fOut, 'b')
    xlim([max(t)/300 max(t)/250])
    ylim([-1.1*max(abs(fOut)) 1.1*max(abs(fOut))])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Time, s')
    ylabel('Amplitude')
    title('The signal in the time domain')
    %}
    
    %%{
    figure;
    i = find(max(t)/300 <t & t < max(t)/250 ); 
    t = t(i);
    x = fOut(i);
      
    plot(t, x,'b')
     
    hold on
    [pks, locs] =findpeaks(x);
    s = 150;
    c = linspace(1,10, length(pks)); 
    scatter(t(locs), pks,[],c); 
    %legend('peaks')
    xlim([min(t) max(t)])
    ylim([-1.1*max(abs(fOut)) 1.1*max(abs(fOut))])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Time, s')
    ylabel('Amplitude')
    title('The signal in the time domain')
    figure
    findpeaks(x);
    %}
    
end

if PLOT3 == 1
    figure,
    fy=fft(fOut);
    L=length(fOut);
    L2=round(L/2);
    fa=abs(fy(1:L2)); % half is essential, rest is aliasing
    
    fmax=Fs/2; % maximal frequency
    fq=((0:L2-1)/L2)*fmax; % frequencies
    plot(fq,fa);
    [pks, locs] = findpeaks(fa,'MinPeakProminence',fmax/1000);
    plot(locs,pks)
    return
end

% p = audioplayer(fOut, fs);
% p.play;
% figure;
% stem(1:N, fOut(:));
% title(' Channel');

df = Fs / N;
w = (-(N/2):(N/2)-1)*df;
y = fft(fOut(:), N) / N; % For normalizing, but not needed for our analysis
y2 = fftshift(y);

%{
figure;
plot(w,abs(y2));
%}

y2=y2';
temp = 1; 

 y_f = zeros(1,2000);
  for j=1:N                             %size(w) == size(N) -- Judy
     if  w(1,j) > 17000 &&  w(1,j) <= 19000  %band filter ? -- Judy
        y_f(1,temp) =  abs(y2(1,j));
        temp = temp+1;
     end
  end
%size(y_f)

if PLOT4 == 1
    figure,
    histogram(y_f)
    xlim([-1.1*max(abs(y_f)) 1.1*max(abs(y_f))])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Signal amplitude')
    ylabel('Number of samples')
    title('Probability distribution of the signal')
end