fd = fopen('out.txt');
C = textscan(fd, '%f');
fclose(fd);
C = C{1};
C = reshape(C, [3 10]);

p1 = C(2,1:2:end);
p2 = C(2,2:2:end);

r1 = C(3,1:2:end);
r2 = C(3,2:2:end);


figure
hold on
grid on
plot(p1, r1, 'o-')
plot(p2, r2, 'o-')
xlabel('Precision')
ylabel('Recall')

legend('Prewitt', 'DoG 7x7')
