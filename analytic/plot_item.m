 unix('cd dat; source ~/.bash_profile; ossget.sh itm.txt')
% filename1 = 'met.good.0116.txt'
% filename2 = 'stp.good.0116.txt'
datpath = 'dat/';
figpath = 'fig/';
% 20 + 100 * 14 + 1
filename1 = strcat(datpath , 'itm.txt');
a = load(filename1);

close(figure(4))
figure(4)
%             line += ' %d %f %f %f %f %f %f' % (
%             item.x['id'], item.x['t'] - item.x['t0'], item.x['q'], item.x['dq'], item.x['ctr'],
%             item.x['theta_lower_ctr'], item.x['theta_higher_ctr'])
%             for k in range(int(self.param['im_feature_dim'])):
%                 line += ' %f' % (item.x['feature_' + str(k)])
%         self.fp_record_item.write(line + '\n')
strlegend=[{'id'}, {'dt'}, {'q'}, {'dq'}, {'ctr'}, {'l_ctr'}, {'h_ctr'}, {'stage'}, {'f1'},{'f2'},{'f3'},{'f4'},{'f5'},{'f6'},{'f7'}]

subplot(2,2,1)
for kk = 1 : 12
for k = 1 + (2:15)+ kk * 10 * 15
%     plot(a(:,1), log(a(:,k)) + (mod(k, 14) + 1) * 3)
%     plot(a(:,1), log(a(:,k)))
    plot(a(:, 1), a(:, k))
    ylabel(strlegend(mod(k-2, 15)+1))
    pause
end
end


% legend(strlegend(2:15))

% ylabel('log')
