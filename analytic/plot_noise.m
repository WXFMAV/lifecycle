unix('source ~/.bash_profile; ossget.sh noise.txt')

f4 = fopen('noise.txt','r');
noise = textscan(f4, '[%f %f %f]');
% noise = textscan(f4, '[%f %f]');
fclose(f4);

close(figure(3))
sz=length(noise{1});
xspace = [1 : sz];
figure(3)
% plot(xspace, [noise{1}';noise{2}';noise{3}'])
plot(xspace, [noise{1}'; noise{2}'])
