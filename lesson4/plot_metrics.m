fd = fopen('out.txt');
C = textscan(fd, '%f');
fclose(fd);
C = C{1};
C = reshape(C, 8, []);

% TP = C(2,:);
% FP = C(3,:);
% FN = C(4,:);
% TN = C(5,:);

% Recall      = TP./(TP+FN);
% Precision   = TP./(TP+FP);
% Specificity = TN./(TN+FP);
% FPR         = FP./(TN+FP);
% FNR         = FN./(TP+FN);
% PofWC       = 100*(FN+FP)./(TP+FN+FP+TN);
% Fm          = 2*(Precision.*Recall)./(Precision+Recall);


Recall      = C(1,:);
Precision   = C(2,:);
Specificity = C(3,:);
FPR         = C(4,:);
FNR         = C(5,:);
PofWC       = C(6,:);
Fm          = C(7,:);


figure
plot(Recall)

figure
plot(Precision)

figure
plot(Specificity)

figure
plot(FPR)

figure
plot(FNR)

figure
plot(PofWC)

figure
plot(Fm)
