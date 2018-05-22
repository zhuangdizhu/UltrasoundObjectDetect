close all
% amp=10 
% fs=20500  % sampling frequency
% duration=3
% freq=30000
% values=0:1/fs:duration;
% a=amp*sin(2*pi* freq*values);
% sound(a);
% plot(a)
Fs = 44100;
toneFreq1 = 17500;  % Tone frequency, in Hertz. must be less than .5 * Fs.
nSeconds = 5;      % Duration of the sound
y1 = sin(linspace(0, nSeconds*toneFreq1*2*pi, round(nSeconds*Fs)));

toneFreq2 = 17700;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y2 = sin(linspace(0, nSeconds*toneFreq2*2*pi, round(nSeconds*Fs)));

toneFreq3 = 17900;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y3 = sin(linspace(0, nSeconds*toneFreq3*2*pi, round(nSeconds*Fs)));

toneFreq4 = 18100;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y4 = sin(linspace(0, nSeconds*toneFreq4*2*pi, round(nSeconds*Fs)));

toneFreq5 = 18300;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y5 = sin(linspace(0, nSeconds*toneFreq5*2*pi, round(nSeconds*Fs)));

toneFreq6 = 18500;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y6 = sin(linspace(0, nSeconds*toneFreq6*2*pi, round(nSeconds*Fs)));

toneFreq7 = 18700;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y7 = sin(linspace(0, nSeconds*toneFreq7*2*pi, round(nSeconds*Fs)));

toneFreq8 = 18900;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y8 = sin(linspace(0, nSeconds*toneFreq8*2*pi, round(nSeconds*Fs)));

toneFreq9 = 19100;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y9 = sin(linspace(0, nSeconds*toneFreq9*2*pi, round(nSeconds*Fs)));

toneFreq10 = 19300;  % Tone frequency, in Hertz. must be less than .5 * Fs.
y10 = sin(linspace(0, nSeconds*toneFreq10*2*pi, round(nSeconds*Fs)));


% toneFreq1 = 17500;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% nSeconds = 5;      % Duration of the sound
% y1 = sin(linspace(0, nSeconds*toneFreq1*2*pi, round(nSeconds*Fs)));
% 
% toneFreq2 = 17700;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y2 = sin(linspace(0, nSeconds*toneFreq2*2*pi, round(nSeconds*Fs)));
% 
% toneFreq3 = 17900;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y3 = sin(linspace(0, nSeconds*toneFreq3*2*pi, round(nSeconds*Fs)));
% 
% toneFreq4 = 18100;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y4 = sin(linspace(0, nSeconds*toneFreq4*2*pi, round(nSeconds*Fs)));
% 
% toneFreq5 = 18300;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y5 = sin(linspace(0, nSeconds*toneFreq5*2*pi, round(nSeconds*Fs)));
% 
% toneFreq6 = 18500;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y6 = sin(linspace(0, nSeconds*toneFreq6*2*pi, round(nSeconds*Fs)));
% 
% toneFreq7 = 18700;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y7 = sin(linspace(0, nSeconds*toneFreq7*2*pi, round(nSeconds*Fs)));
% 
% toneFreq8 = 18900;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y8 = sin(linspace(0, nSeconds*toneFreq8*2*pi, round(nSeconds*Fs)));
% 
% toneFreq9 = 19100;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y9 = sin(linspace(0, nSeconds*toneFreq9*2*pi, round(nSeconds*Fs)));
% 
% toneFreq10 = 18300;  % Tone frequency, in Hertz. must be less than .5 * Fs.
% y10 = sin(linspace(0, nSeconds*toneFreq10*2*pi, round(nSeconds*Fs)));

y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10;

%sound(y,Fs); % Play sound at sampling rate Fs
%plot(y);
size(y)
%psd(y)
%spectrogram()
figure
plot(y1)

figure
 fy=fft(y);
 L=length(y);
 L2=round(L/2);
 fa=abs(fy(1:L2)); % half is essential, rest is aliasing
 fmax=Fs/2; % maximal frequency
 fq=((0:L2-1)/L2)*fmax; % frequencies
 size(fq)
 plot(fq,fa);
 filename = '../../Data/tone1.wav';
 
 %wavwrite( y, Fs, 32, filename);
 audiowrite(filename, y, Fs, 'BitsPerSample', 32);