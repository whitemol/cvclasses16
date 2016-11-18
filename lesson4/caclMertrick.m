v_res = VideoReader('dd73_res.avi');
result = VideoReader('result.avi');

Rec = [];
Prec = [];
Spec = [];
FPR = [];
FNR = [];
PWC = [];
Fm = [];

count = 0;
while hasFrame(v_res) && hasFrame(result)
    frame_res = rgb2gray(readFrame(v_res));
    frame_result = rgb2gray(readFrame(result));
    
    frame_res = frame_res > 240;
    frame_result = frame_result > 240;
    
    TP = sum(sum(frame_res & frame_result));
    TN = sum(sum(~frame_res & ~frame_result));
    FN = sum(sum(~frame_res & frame_result));
    FP = sum(sum(frame_res & ~frame_result));
    
    if count > 30*3
        Rec = [Rec, TP / (TP + FN)];
        Prec =[Prec, TP / (TP + FP)];
        Spec =[Spec, TN / (TN + FP)];
        FPR =[FPR, FP / (TN + FP)];
        FNR =[FNR, FN / (TP + FN)];
        PWC =[PWC, 100 * (FN+FP) / (TP+TN+FP+FN)];
        Fm =[Fm, 2 * (Prec(end) * Rec(end)) / (Prec(end) + Rec(end))];
    end
    count =  count + 1;
end

figure; hold on; title('Rec'); plot(Rec);
figure; hold on; title('Prec'); plot(Prec);
figure; hold on; title('Spec'); plot(Spec);
figure; hold on; title('FPR'); plot(FPR);
figure; hold on; title('FNR'); plot(FNR);
figure; hold on; title('PWC'); plot(PWC);
figure; hold on; title('Fm'); plot(Fm);
