function Ret = feature_vector_time_domain(filename, label) 
if nargin < 2
    label = 1;
    filename='../Data/onbox/zhuangdi/plastic_box_0318/plastic_box100.wav';
    %filename='/Users/judy/Projects/UltraSoundObjectDetection/Data/onbox/salman/book/2018-Mar-12_10-10-43.wav';
end

[y,fs] = audioread(filename);
channel_n = size(y,2);
%% Design a bandpass filter that filters out between 700 to 12000 Hz
n = 7;
beginFreq = 17000 / (fs/2);
endFreq = 19000 / (fs/2);
[b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); %btype, analog -- Judy
x = filter(b, a, y);

x = x(1: round(0.5 * size(x,1)), :);
%% Add fine-grained features using a sliding window
features = add_sliding_window_features(x, channel_n); 

Ret = [features label];   

end

function M = add_features(x) 
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
        features = [features signal_feature];
    end
    %signal_feature = add_features2(chunk_x); 
end
%} 
end
 