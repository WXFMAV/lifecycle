close(figure(1))
fid = fopen('met.txt','r');
m = textscan(fid, 'eps:%f item_count:%f item_new:%f item_dead:%f id:%f t:%f q:%f ctr:%f ctr_avg:%f Q_all:%f startup_count:%f maturity_count:%f decline_count:%f startup_score:%f maturity_score:%f decline_score:%f');
% eps: 1, item_count: 2, item_new: 3, item_dead: 4, 
% id : 5, t : 6, q : 7, ctr : 8, ctr_avg : 9, q_all : 10
% startup_count : 11, maturity_count : 12, decline_count : 13
% startup_score : 14, maturity_score : 15, decline_score : 16
f2 = fopen('obs.txt', 'r');
o = textscan(f2, 'eps: %f %f %f %f %f %f %f None');
% % eps: 
%         x[2] = np.mean(s_x[:, 1]) # ??ctr
%         x[3] = np.mean(s_x[:, 2]) # ????
%         x[4] = np.std(s_x[:, 1])  # ctr????
%         x[5] = np.mean(s_x[:, 3]) # ????
%         x[6] = np.mean(s_x[:, 0]) # ????????
%          7 = reward

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
p1 = plot(o{1}, o{2}, 'r');
hold on
p2 = plot(o{1}, o{4}, 'b');
n = length(o{1});
axis([0, n, 0, 0.1])
xlabel('time')
ylabel('ctr')
title('average ctr, std ctr')
grid on
legend('average ctr', 'std ctr')

subplot(4,2,5)
p1 = plot(o{1}, o{5}, 'r');
n = length(o{1});
hold on
plot(m{1}, m{14}, 'b');
plot(m{1}, m{15}, 'g');
plot(m{1}, m{16}, 'k');
axis([0, n, 0, 1.0])
xlabel('time')
ylabel('percent')
title('percentage')
legend('p.c. of new item', 'avg score of new item', 'avg score of maturity', 'avg score of decline')
grid on

subplot(4,2,6)
p1 = plot(o{1}, o{7}, 'r');
n = length(o{1});
axis([0, n, min(o{7}), max(o{7})])
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