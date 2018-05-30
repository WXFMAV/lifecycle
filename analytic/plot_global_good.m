% unix('source ~/.bash_profile; ossget.sh met.txt stp.txt')
filename1 = 'met.good.0116.txt'
filename2 = 'stp.good.0116.txt'
% filename1 = 'met.txt'
% filename2 = 'stp.txt'

f1 = fopen(filename1,'r');
m = textscan(f1, 'eps:%f item_count:%f item_new:%f item_dead:%f id:%f t:%f q:%f ctr:%f ctr_avg:%f Q_all:%f startup_count:%f maturity_count:%f decline_count:%f startup_score:%f maturity_score:%f decline_score:%f reward:%f info:None  x0:%f x1:%f x2:%f x3:%f x4:%f');
%                 eps:%f item_count:%f item_new:%f item_dead:%f id:%f t:%f q:%f ctr:%f ctr_avg:%f Q_all:%f startup_count:%f maturity_count:%f decline_count:%f startup_score:%f maturity_score:%f decline_score:%f reward:%f info:None
fclose(f1);

% eps: 1, item_count: 2, item_new: 3, item_dead: 4, 
% id : 5, t : 6, q : 7, ctr : 8, ctr_avg : 9, q_all : 10
% startup_count : 11, maturity_count : 12, decline_count : 13
% startup_score : 14, maturity_score : 15, decline_score : 16
% reward : 17
% x0 : 18, x1 : 19, x2 : 20, x3 : 21, x4 : 22
%         x[0] = np.mean(s_x[:, 1]) # ??ctr
%         x[1] = np.mean(s_x[:, 2]) # ????
%         x[2] = np.std(s_x[:, 1])  # ctr????
%         x[3] = np.mean(s_x[:, 3]) # ????
%         x[4] = np.mean(s_x[:, 0]) # ????????

f2 = fopen(filename2, 'r');
s = textscan(f2, 'eps:%f action:[%f  %f %f %f] reward:%f info:None state:[  %f   %f   %f   %f   %f]');
fclose(f2);

% eps : 1
% action : 2, 3, 4, 5
% reward : 6
% state :  7, 8, 9, 10, 11

close(figure(1))
close(figure(2))

figure(1)

subplot(4,2,1)
p1=plot(m{1}, m{3},'r');
hold on
p2=plot(m{1}, m{4},'b');
n = length(m{1});
axis([0 n 0 30])
xlabel('time')
ylabel('item count')
title('new item/dead item')
legend('new', 'dead')


subplot(4,2,2)
p1= plot(m{1}, m{9},'b');
n = length(m{1});
hold on
p2 = plot(m{1}, m{8}, 'r');
axis([0 n 0 0.2])
xlabel('time')
ylabel('ctr avg')
title('ctr avg')
legend('average', ['id=', num2str(m{5}(1))])
grid on

subplot(4,2,3)

p1 = plot(m{1}, m{10}, 'b');
n = length(m{1});
axis([0 n 0 max(m{10})])
xlabel('time')
ylabel('Q all')
title('q all')
grid on

subplot(4,2,4)
p1 = plot(m{1}, m{18}, 'r');
hold on
p2 = plot(m{1}, m{20}, 'b');
n = length(m{1});
axis([0, n, 0, 0.1])
xlabel('time')
ylabel('ctr')
title('average ctr, std ctr')
grid on
legend('average ctr', 'std ctr')

subplot(4,2,5)
plot(m{1}, m{14}, 'b');
hold on;
plot(m{1}, m{15}, 'g');
plot(m{1}, m{16}, 'k');
axis([0, n, 0, 1.0])
xlabel('time')
ylabel('percent')
title('percentage')
legend('avg score of new item', 'avg score of maturity', 'avg score of decline')
grid on

subplot(4,2,6)
p1 = plot(m{1}, m{17}, 'r');
n = length(m{1});
axis([0, n, min(m{17}), max(m{17})])
xlabel('time')
ylabel('reward')
title('reward')
grid on

subplot(4,2,7)
p1 = plot(m{1}, m{2}, 'r');
n = length(m{1});
% axis([0, n, min(m{2}), max(m{2})])
axis([0, n, 0, 13000])
xlabel('time')
ylabel('item count')
title('item count')
grid on

subplot(4,2, 8)
p1 = plot(m{1}, m{11}, 'r');
hold on;
p2 = plot(m{1}, m{12}, 'g');
p3 = plot(m{1}, m{13}, 'b');
xlabel('time')
ylabel('count')
title('stage count')
legend('startup', 'maturity', 'decline')
saveas(gcf, 'metrics.jpg', 'jpg')

figure(2)
subplot(2,2,1)
plot(s{1}, s{2})
hold on;
plot(s{1}, s{3})
plot(s{1}, s{4})
plot(s{1}, s{5})
legend('1','2','3','4')
grid on
xlabel('time')
ylabel('scalar')
title('action')


subplot(2,2,2)
plot(s{1}, s{7})
hold on;
 plot(s{1}, s{8} / min(s{8}) - 1.0)
plot(s{1}, s{9})
plot(s{1}, s{10})
plot(s{1}, s{11} / min(s{11}) - 1.0 )
legend('1','2','3','4','5')
grid on
xlabel('time')
ylabel('scalar')
title('state')

saveas(gcf, 'action.jpg', 'jpg')
