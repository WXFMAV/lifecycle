
y(1) = 0.05
ch = 0.15
cl = 0.05
t2 = 60
t1 = 30

for k = 2:100
    x = (k - 50)/10
    y(k) = y(k-1) - exp(-x)/(1+exp(-x))^2 * (ch-cl);
end

close(figure(1))
figure(1)
plot(y)