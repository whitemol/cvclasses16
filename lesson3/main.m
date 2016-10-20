close all; clear all; clc;

img_gray = rgb2gray(imread('dd73_512.jpg'));
img_edge_men = imread('dd73_512_edge.bmp');

PrecV = [];
RecV = [];

figure(1); hold on; grid on;
title('Recall / Precision');
xlabel('Precision'); ylabel('Recall');
for th = 0:10:100
    img_edge = edge_threshold(img_gray, th, 'RobCross');
    [Prec, Rec] = PrecRec(img_edge_men, img_edge);
    PrecV = [PrecV, Prec];
    RecV = [RecV, Rec];
    plot(Prec, Rec, '*r');
    if th == 100
        figure(2); imshow(img_edge);
        figure(1); plot(Prec, Rec, 'og');
        plot(PrecV, RecV, 'r');
    end
end

PrecV = [];
RecV = [];

figure(1);
for th = 0:10:100
    img_edge = edge_threshold(img_gray, th, 'LoG');
    [Prec, Rec] = PrecRec(img_edge_men, img_edge);
    PrecV = [PrecV, Prec];
    RecV = [RecV, Rec];
    plot(Prec, Rec, '*b');
        if th == 100
            figure(3); imshow(img_edge);
            figure(1); plot(Prec, Rec, 'og');
            plot(PrecV, RecV, 'b');
    end
end
    