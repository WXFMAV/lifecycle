% unix('cd dat; source ~/.bash_profile; ossget.sh met.txt stp.txt stt.txt')
% datpath = 'dat/param_high_dim1d/';
datpath = 'dat/param_exp_cnn/';
figpath = 'fig/';

filename1 = strcat(datpath , 'met.txt');
m = load(filename1);

% 1 line = '%f' % (t) 
% 2 line += ' %d' % (len(self.item_list))  total count
% 3 line += ' %d' % ((self.item_new_count))
% 4 line += ' %d' % ((self.item_dead_count))
% 5 line += ' %d %f %f %f' % (i_id, i_t, i_q, i_ctr)
% 9 line += ' %f' % (s_ctr/s_cnt)
% 10 line += ' %f' % (s_Q)
% 11 line += ' %f' % (reward)
% 12 line += ' %s' % (str(0.0))
% 13 line += ' %f %f %f %f %f' % (x[0], x[1], x[2], x[3], x[4])
% 18,5x5, for j in range(0, len(s_stage_score)):
%     line += ' %f %f %f %f %f' % (s_stage_cnt[j], s_stage_score[j], s_stage_dq[j], s_stage_q[j], s_stage_ctr[j])
% 43 line += ' %f' % (s2 / s_cnt) #global click
% 44 grow up cost time
% #print(line)
% self.fp_record_metrics.write(line + '\n')

close(figure(2))
figure(2)
subplot(4,2,1)
plot(m(:,1), m(:, 11), 'r')
xlabel('time')
ylabel('reward')
title('reward')

subplot(4,2,2)
plot(m(:, 1), m(:, 9), 'b')
hold on
plot(m(:, 1), m(:, 8), 'g')
xlabel('time')
ylabel('avg ctr')
title('ctr')
legend('avg ctr', 'ctr example')

subplot(4,2,3)
plot(m(:, 1), m(:, 24),'-')
hold on
plot(m(:, 1), m(:, 29), '-')
plot(m(:, 1), m(:, 34), '-')
plot(m(:, 1), m(:, 39), '-')
legend('startup', 'growth',  'maturity', 'decline')
xlabel('time')
ylabel('score')
title('score')

subplot(4,2,4)
plot(m(:, 1), m(:, 23) ./ m(:, 2), '-')
hold on 
plot(m(:, 1), m(:, 28) ./ m(:, 2), '-')
plot(m(:, 1), m(:, 33) ./ m(:, 2), '-')
plot(m(:, 1), m(:, 38) ./ m(:, 2), '-')
legend('startup', 'growth',  'maturity', 'decline')
xlabel('time')
ylabel('qunatity')
title('items in different stage')

subplot(4,2,5)
plot(m(:, 1), m(:, 25), '-')
hold on 
plot(m(:, 1), m(:, 30), '-')
plot(m(:, 1), m(:, 35), '-')
plot(m(:, 1), m(:, 40), '-')
legend('startup', 'growth', 'maturity', 'decline')
xlabel('time')
ylabel('dq')
title('avg dq in different stage')

subplot(4,2,6)
plot(m(:, 1), m(:, 27), '-')
hold on 
plot(m(:, 1), m(:, 32), '-')
plot(m(:, 1), m(:, 37), '-')
plot(m(:, 1), m(:, 42), '-')
legend('startup', 'growth',  'maturity', 'decline')
xlabel('time')
ylabel('ctr')
axis([min(m(:,1)), max(m(:,1)), 0, 0.2])
title('avg ctr in different stage')


subplot(4,2,7)
plot(m(:, 1), m(:, 43), '-')
xlabel('time')
ylabel('global click')
title('global click')

subplot(4,2,8)
plot(m(:, 1), m(:, 44), '-')
xlabel('time')
ylabel('avg grow up cost time')
title('grow up cost time')
