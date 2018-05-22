function Ret = feature_vector1(filename, label) 
DEBUG1 = 0; 
if nargin < 2
    filename='../Data/wood/2018-Feb-21_11-16-59.wav';
    label = 1;
end

[y,fs] = audioread(filename);

%% Design a bandpass filter that filters out between 700 to 12000 Hz
n = 7;
beginFreq = 17000 / (fs/2);
endFreq = 19000 / (fs/2);
[b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); %btype, analog -- Judy
x = filter(b, a, y);
N = length(x);                  % signal length
 
%% extract general features 
[pks, locs, w, p] = findpeaks(x);                      % x is the amplitude

%${
window = hanning(N, 'periodic');
figure
%periodogram(x, window, N, fs, 'power');
 
[pxx, ~] = periodogram(x, window, N, fs, 'power');
 pxx  = 10*log10(pxx);  
plot(pxx)
[psd_pks, psd_locs, psd_w, psd_p] = findpeaks(pxx);    % pxx is the psd
%}

%%{
%f11 = add_salman_features(x, DEBUG1);             % dim = 4
%f13 = add_judy_features(pks, locs, w, p);         % dim = 5

%f12 = add_salman_features(pxx, DEBUG1);
%f14 = add_judy_features(psd_pks, psd_locs, w, p);
%Ret = [f11 label]; 
%}
 

%%{
%% Add fine-grained features using a sliding window
f21 = add_sliding_window_features(x, pks, locs);
%f22 = add_sliding_window_features(pxx, psd_pks, psd_locs);
 Ret = [f21 label];
%}
%Ret = [f11 f12 f13 f21 f22 label];  
end

function M1 = add_salman_features(x, DEBUG1)
if nargin < 2
    DEBUG1 = 0;
end
%size(x)
%% compute and display the minimum and maximum values
maxval = max(x);
minval = min(x);

%% compute and display the the DC and RMS values
u = mean(x);
s = std(x);
 
%m = median(x);
%%{
%% compute and display the dynamic range
D = 20*log10(maxval/min(abs(nonzeros(x))));

%% compute and display the crest factor
Q = 20*log10(maxval/s);
%% compute and display the autocorrelation time
[Rx, ~] = xcorr(x, 'coeff');
ind = find(Rx>0.05, 1, 'last');
N = length(x);
%RT = (ind-N)/fs;

if DEBUG1 == 1 
    disp(['Max value = ' num2str(maxval)])
    disp(['Min value = ' num2str(minval)])

    disp(['Mean value = ' num2str(u)])
    disp(['RMS value = ' num2str(s)])
    
    disp(['Dynamic range D = ' num2str(D) ' dB'])
    disp(['Crest factor Q = ' num2str(Q) ' dB'])
    %disp(['Autocorrelation time = ' num2str(RT) ' s'])
end
%}
M1 = [maxval minval u s];

end

function M2 = add_judy_features(pks, locs, w, p) 
len = length(pks);

%% compute and display the Mean Distance & Standard Diviation of Peaks
distance = locs(2:len) - locs(1:len-1);
avgD = mean(distance);
stdD = std(distance);
%medianD = median(distance);
%% compute and display the Mean Amplitude of Peaks
avgM = mean(pks);
stdM = std(pks);
%medianM = median(pks);

%{
avgW = mean(w);
stdW = std(w);

avgP = mean(p);
stdP = std(p);

M2 = [1/len avgD stdD  avgM, stdM avgW, stdW, avgP, stdP];
%}
M2 = [1/len avgD stdD  avgM, stdM];
 
end


function fetures = add_sliding_window_features(x, pks, locs)
%{
chunk = 30;
n = round(length(pks)/2);
n = length(pks);
chunk_size = round(n/chunk);
pks = pks(1: n);
locs = locs(1:n);
fetures = [];
for i = 1:chunk
    start_i = (i-1)*chunk_size + 1;
    end_i = min(n, i*chunk_size);
    chunk_pks = pks(start_i:end_i);
    chunk_locs = locs(start_i:end_i);
    %signal_feature1 = add_salman_features(x, 0);
    signal_feature2 = add_judy_features(chunk_pks,chunk_locs);
    %fetures = [fetures signal_feature1 signal_feature2];
    fetures = [fetures signal_feature2];
end
%}

%%{
chunk = 30; 
n = length(x);
chunk_size = round(n/chunk); 
fetures = []; 
for i = 1:chunk
    start_i = (i-1)*chunk_size + 1;
    end_i = min(n, i*chunk_size);
    chunk_x = x(start_i:end_i);
    [chunk_pks, chunk_locs] = findpeaks(chunk_x); 
    signal_feature2 = add_judy_features(chunk_pks,chunk_locs); 
    fetures = [fetures signal_feature2];
end
%}
end
 