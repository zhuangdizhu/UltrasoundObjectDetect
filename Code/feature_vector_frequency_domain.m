function Ret = feature_vector_frequency_domain(filename, label) 
if nargin < 2
    label = 1;
    %filename = '/Users/judy/Projects/UltraSoundObjectDetection/Data/inbox/static/empty/static_inbox_empty99.wav';
    filename = '/Users/judy/Projects/UltraSoundObjectDetection/Data/inbox/moving/empty/moving_empty30.wav';
    %filename = '/Users/judy/Projects/UltraSoundObjectDetection/Data/inbox/moving/book/moving_book15.wav';
end
close all
figure
[y,fs] = audioread(filename);
channel_n = size(y,2);


%% Design a bandpass filter that filters out between 700 to 12000 Hz
n = 7;
beginFreq = 17000 / (fs/2);
endFreq = 19000 / (fs/2);
[b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); %btype, analog -- Judy
x = filter(b, a, y);  
N = length(x);
 
 
pxx = get_pwelch(x); 
%{
for i = 1:1
    figure
    %spectrogram(x(:,i),'yaxis')
    spectrogram(x(:,i),1024, 512);
   % r = SpectralSlope(s, fs);
  %  size(r)
end
%}
   
%% Add fine-grained features using a sliding window
%features = add_sliding_window_features(pxx, channel_n); 
features = add_psd_features(pxx); 
Ret = [features label];   

end

function features = add_psd_features(pxx) 
pxx = resample(pxx, 1,200); 
pxx = pxx';
features = reshape(pxx, 1, size(pxx,1)*size(pxx,2));
end
function pxx = get_pwelch(x)
%%{
 %% extract frequency features  
  
 pxx = pwelch(x);
 
 pxx  = 10*log10(pxx); 
 NF = size(pxx,1); 
 range = floor(NF*0.65) : floor(NF * 0.9);
 pxx = pxx(range,:);   
%}
end
 
function M = add_features2(x)
  
%% compute and display the minimum and maximum values
maxval = max(x);
minval = min(x);

%% compute and display the the DC and RMS values
u = mean(x);
s = std(x);  
M = [maxval, minval, u s];

end

function phases = get_phase(x, fs, channel_n) 
phases = [];
for i = 1:channel_n
    current_channel_x = x(:,i);  
    s = fft(current_channel_x); 
    
    phs = angle(fftshift(s));
    phs = phs/pi;
    
    ls = length(s);
    f = (-ls/2:ls/2-1)/ls*fs;
 
    
    r = 10;
    phs = resample(phs, 1,r); 
    f = resample(f, 1,r);
    
    
    lf = length(f);
    s = round(0.8*lf);
    e = round(0.95*lf);
     
     %{
    figure 
    plot(f(s:e), phs(s:e))
    xlabel 'Frequency (Hz)'
    ylabel 'Phase / \pi'
    grid 
    %}
    phs = phs(s:e);  
    phases = [phases phs];
end  
end

function M = add_features(x ) 
[pks, locs] = findpeaks(x); 
len = length(pks);

%% compute and display the Mean Distance & Standard Deviation of Peaks
distance = locs(2:len) - locs(1:len-1);
avgD = mean(distance);
stdD = std(distance); 
%% compute and display the Mean & Standard Deviation of Peak Amplitudes
avgM = mean(pks);
stdM = std(pks); 
M = [1/len avgD stdD  avgM, stdM];
end
 
function [vssl] = SpectralSlope (X, f_s)
% @param X: spectrogram (dimension FFTLength X Observations)
% @param f_s: sample rate of audio data (unused)
%
% @retval vsk spectral slope
    % compute mean
    mu_x    = mean(abs(X), 1);
    
    % compute index vector
    kmu     = [0:size(X,1)-1] - size(X,1)/2;
    
    % compute slope
    X       = X - repmat(mu_x, size(X,1), 1);
    vssl    = (kmu*X)/(kmu*kmu');
end

function features = add_sliding_window_features(x, channel_n)
%%{
chunk = 30; 
n = size(x,1);
chunk_size = round(n/chunk); 
features = []; 
for j = 1:channel_n
    x_current_channel = x(:,j);
    for i = 1:chunk
        start_i = (i-1)*chunk_size + 1;
        end_i = min(n, i*chunk_size);
        chunk_x_current_channel = x_current_channel(start_i:end_i);
    
        signal_feature = add_features(chunk_x_current_channel); 
        %signal_feature = add_features2(chunk_x_current_channel);
        features = [features signal_feature];
    end
    %signal_feature = add_features2(chunk_x); 
end
%} 
end
 