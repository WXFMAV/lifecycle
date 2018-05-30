unix('cd dat; source ~/.bash_profile; ossget.sh act.txt')

datpath = 'dat/';
figpath = 'fig/';

% 20 + 100 * 14 + 1
filename1 = strcat(datpath , 'act.txt');
a = load(filename1);

close(figure(5))
figure(5)
hold on

for k = 2:15
    plot(a(:, 1), a(:, k) + k * 3, '.')
    line([min(a(:, 1)), max(a(:, 1))], [k * 3 - 1, k * 3 - 1])
    line([min(a(:, 1)), max(a(:, 1))], [k * 3 + 1, k * 3 + 1])
    
end

title('action')

