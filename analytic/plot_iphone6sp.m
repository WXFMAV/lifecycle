% a=load('dat/iphone_6s_plus_sales.txt');
% close(figure(1));
% figure(1);
% plot(a(:,5));
% figure(1);

% a=load('dat/lc_item_deal_log_analyse.20180402.txt');
% close(figure(1));
% figure(1);
% plot(a(:,5));

% a=load('dat/lc_plot_xy.20180402.txt');
a=load('dat/lc_plot_xy.20170101_ipv.txt');
for k = 1: 1000
    close(figure(1));
    h=figure(1);
    idx = find(a(:,1)==k);
%     pk=6;
%     pk=8;
%     sub = find(a(idx, pk) < mean(a(idx, pk) * 5));
%     plot(a(idx(sub),4), a(idx(sub), pk));
%     saveas(h, strcat('lifecycle_fig/timelong_ipv_8_',num2str(k, '%03d'),'.png'));
    pk=8;
    sub = find(a(idx, 6) ./ (0.0001+a(idx, 8)) < mean(a(idx, 6) ./ (0.0001+a(idx, 8)) * 5));
    plot(a(idx(sub),4), a(idx(sub), 6) ./ (0.0001+a(idx(sub), 8)));
    saveas(h, strcat('lifecycle_fig/timelong_cvr_68_',num2str(k, '%03d'),'.png'));

end
