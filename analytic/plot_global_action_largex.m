close(figure(2))
f3 = fopen('stp.txt', 'r');
s = textscan(f3, 'eps:%f state:[  %f   %f   %f   %f   %f] action:[%f  %f %f %f] reward:%f info:None');
%s = textscan(f3, 'eps:%f state:[   %f   %f   %f ...,   %f   %f   %f] action:[%f  %f %f %f] reward:%f info:None');
%   %f   %f   %f ...,   %f   %f   %f
% time : 1
% state :  2, 3, 4, 5, 6
% action : 7, 8, 9, 10
% reward : 11
%'eps:474.000000 state:[  6.85414751e-02   3.01725485e+05   3.26665696e-02   8.68343054e-01   3.66991108e+02] action:[-1.69894069  0.65535664 -1.14696866 -0.97209399] reward:405.479738 info:None'

figure(2)
subplot(2,2,1)
plot(s{1}, s{7})
hold on;
plot(s{1}, s{8})
plot(s{1}, s{9})
plot(s{1}, s{10})
legend('1','2','3','4')
grid on
xlabel('time')
ylabel('scalar')
title('action')


subplot(2,2,2)
plot(s{1}, s{2})
hold on;
 plot(s{1}, s{3} / min(s{3}) - 1.0)
plot(s{1}, s{4})
plot(s{1}, s{5})
plot(s{1}, s{6} / min(s{6}) - 1.0 )
legend('1','2','3','4','5')
grid on
xlabel('time')
ylabel('scalar')
title('state')

saveas(gcf, 'action.jpg', 'jpg')
