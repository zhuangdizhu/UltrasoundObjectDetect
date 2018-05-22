function extract_features()
%% Variables
output_filename = 'Feature_0320_static_empty3.csv'; 
output_folder = '/Users/judy/Projects/UltraSoundObjectDetection/Features/';  

%%{
%% Open data folder 
cd /Users/judy/Projects/UltraSoundObjectDetection/Data/inbox/static/empty/
data_dir = '/Users/judy/Projects/UltraSoundObjectDetection/Data/inbox/static/empty/';
label = 9; % set class label
f = dir('*.wav');
N = length(f); % row
filename = f(1).name;
cd /Users/judy/Projects/UltraSoundObjectDetection/Code/

%% Extract feature vectors for each sample
M = length(feature_vector_frequency_domain([data_dir filename], label) ); % column
%M = length(feature_vector_time_domain([data_dir filename], label) ); % column
dataset = zeros(N,M);
for i = 1:N
    filename = f(i).name;
   % vectors = feature_vector_time_domain([data_dir filename], label);
    vectors = feature_vector_frequency_domain([data_dir filename], label);
     
    dataset(i,:) = vectors;
end
%}

%dataset = [dataset1; dataset];
csvwrite([output_folder output_filename], dataset);
end