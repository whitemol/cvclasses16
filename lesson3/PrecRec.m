function [Prec Rec] = PrecRec(orig, test)

TP = sum(orig(:) & test(:));
T = sum(orig(:));

TN = sum(~orig(:) & ~test(:));

Prec = TP / (TP + TN);

Rec = TP / T;

end