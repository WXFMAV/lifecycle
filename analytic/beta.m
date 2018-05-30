N=8

for h=1:N
    Dx=h * (N - h + 1) / (N+1)^2/(N+2);
    [h, Dx]
end