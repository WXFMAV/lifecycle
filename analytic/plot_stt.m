% unix('source ~/.bash_profile; ossget.sh stt.txt')

filename = 'stt.txt'
% a = load(filename);
sz = size(a)


k = 400;
b = zeros(120, 120);
l = 0;
for i = 1: 120
for j = 1: 120
    l = l + 1;
    if l < sz(2)
        b(i,j) = (a(k, l) + 1) / 2.0;
    end
end
end
figure(1)
imshow(b)

figure(2)
plot(a(k, 7 + (0:999) * 14),'.b')
