% a=load('dat/data.txt');
a=load('dat/data.20180402.cate_level1_1625.txt');
figure(1)
yk = 6;

plot3(log10(a(:,1)), log10(a(:,2)), a(:,yk),'.')
xlabel('x');
ylabel('y');
zlabel('z');
figure(1)
% plot(a(:,2), a(:,3),'.')

N = 100;
s_ctr = zeros(N, 1);
c_ctr = zeros(N, 1);
avg_ctr = zeros(N, 1);
n = size(a,1);

for k = 1:n
    month = round(a(k,1)/30);
    ctrnow = a(k, yk);
    if month<N && month>=1
        s_ctr(month) = s_ctr(month) + ctrnow;
        c_ctr(month) = c_ctr(month) + 1;
    end
end

for k = 1 : N
    if c_ctr(k) > 0 
        avg_ctr(k) = s_ctr(k) / c_ctr(k);
    end
end

figure(2)
subplot(2,1,1)
plot(log10(a(:,1)), a(:,yk),'.')
title('time');

subplot(2,1,2)
plot(log10(a(:,2)), a(:,yk),'.')
title('pv');

figure(3)
subplot(2,1,1);
plot(1:N, avg_ctr, '-');
title('avg');
xlabel('time/month');
ylabel('ctr');
subplot(2,1,2);
plot(1:N, c_ctr, '-');
title('count');
ylabel('count');
xlabel('time/month');