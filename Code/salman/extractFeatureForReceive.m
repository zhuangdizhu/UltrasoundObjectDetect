%clear, clc, close all
DRAW1 = 0; %% set 1 to draw singals, 0 otherwise. -- Judy
DRAW2 = 0; %% set 1 to draw singals, 0 otherwise. -- Judy
DRAW3 = 0; %% set 1 to draw singals, 0 otherwise. -- Judy
DRAW4 = 1;
% get a section of the sound file

%[x, fs] = audioread('receive.wav');   % load an audio file
%[x, fs] = audioread('../Data/receive.wav');   %% -- Judy 
[x,fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/wood/2018-Feb-21_11-19-13.wav');
[x,fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/wood/2018-Feb-21_11-19-13.wav');
%[y,Fs] = audioread('/Users/judy/Projects/UltraSoundObjectDetection/Data/wood/2018-Feb-21_11-19-05.wav');

x = x(:, 1);                        % get the first channel
N = length(x);                      % signal length
t = (0:N-1)/fs;                     % time vector

if DRAW1 == 1 %% -- Judy
    % plot the signal waveform
    figure
    plot(t, x, 'r')
    xlim([0 max(t)])
    ylim([-1.1*max(abs(x)) 1.1*max(abs(x))])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Time, s')
    ylabel('Amplitude')
    title('The signal in the time domain')
    % plot the signal spectrogram
    figure
    spectrogram(x, 1024, 3/4*1024, [], fs, 'yaxis')
    box on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Time, s')
    ylabel('Frequency, Hz')
    title('Spectrogram of the signal')

    h = colorbar;
    set(h, 'FontName', 'Times New Roman', 'FontSize', 14)
    ylabel(h, 'Magnitude, dB')
    return
end

% spectral analysis
w = hanning(N, 'periodic');

[X, f] = periodogram(x, w, N, fs, 'power');
%figure
%periodogram(x, w, N, fs, 'power');
 
X = 20*log10(sqrt(X)*sqrt(2));

if DRAW4 == 1
    % plot the signal spectrum
    figure
    s = semilogx(f, X, 'r');
    size(s.YData)
    xlim([0 max(f)])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    title('Amplitude spectrum of the signal')
    xlabel('Frequency, Hz')
    ylabel('Magnitude, dB')
    return
end


if DRAW2 == 1 
    % plot the signal histogram
    figure
    histogram(x)
    xlim([-1.1*max(abs(x)) 1.1*max(abs(x))])
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Signal amplitude')
    ylabel('Number of samples')
    title('Probability distribution of the signal')
    return
end


% autocorrelation function estimation
[Rx, lags] = xcorr(x, 'coeff');
d = lags/fs;

if DRAW3 == 1
    % plot the signal autocorrelation function
    figure
    plot(d, Rx, 'r')
    grid on
    xlim([-max(d) max(d)])
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Delay, s')
    ylabel('Autocorrelation coefficient')
    title('Autocorrelation of the signal')
    line([-max(abs(d)) max(abs(d))], [0.05 0.05],...
         'Color', 'k', 'LineWidth', 2, 'LineStyle', '--')
end


% compute and display the minimum and maximum values
maxval = max(x);
minval = min(x);
disp(['Max value = ' num2str(maxval)])
disp(['Min value = ' num2str(minval)])
 
% compute and display the the DC and RMS values
u = mean(x);
s = std(x);
disp(['Mean value = ' num2str(u)])
disp(['RMS value = ' num2str(s)])

% compute and display the dynamic range
D = 20*log10(maxval/min(abs(nonzeros(x))));
disp(['Dynamic range D = ' num2str(D) ' dB'])

% compute and display the crest factor
Q = 20*log10(maxval/s);
disp(['Crest factor Q = ' num2str(Q) ' dB'])

% compute and display the autocorrelation time
ind = find(Rx>0.05, 1, 'last');
RT = (ind-N)/fs;
disp(['Autocorrelation time = ' num2str(RT) ' s'])

commandwindow