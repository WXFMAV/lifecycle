unix('source ~/.bash_profile; ossget.sh model_test.txt')

a = load('model_test.txt');

close(figure(4))
figure(4)

size(a)
plot(a(:, 2), a(:, 9), '.r')
% plot3(a(:,1), a(:,2),a(:,7), '.r')
% hold on;
% plot3(a(:,1), a(:,2),a(:,9), '.b')

return 

close(figure(4))
figure(4)

size(a)
subplot(2,1,1)
plot3(a(:,5), a(:,6),a(:,1), '.r')
hold on;
plot3(a(:,5), a(:,6),a(:,3), '.b')

subplot(2,1,2)
plot3(a(:,5), a(:,6),a(:,2), '.r')
hold on;
plot3(a(:,5), a(:,6),a(:,4), '.b')