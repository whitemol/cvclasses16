function result = edge_threshold(img, threshold, type)

    img = double(img);

    if strcmp(type,'LoG')
        result = LoG(img, threshold);
    elseif strcmp(type,'RobCross')
        result = RobCross(img, threshold);
    else
        error('Unknown type.');
    end

end

function img_edge = LoG(img, threshold)
    LoG5 = [0  0 -1  0  0;
            0 -1 -2 -1  0;
           -1 -2 16 -2 -1;
            0 -1 -2 -1  0;
            0  0 -1  0  0];

    img_edge = conv2(img, LoG5);
    img_edge (img_edge <= threshold) = 0;
    img_edge (img_edge > threshold) = 1;
    img_edge = logical(img_edge(3:514,3:514));
end

function img_edge = RobCross(img, threshold)
    Robertsx = [1, 0;
                0,-1];
    Robertsy = [0,-1;
                1, 0];

    img_edge = conv2(img, Robertsx) + conv2(img, Robertsy);
    img_edge (img_edge <= threshold) = 0;
    img_edge (img_edge > threshold) = 1;
    img_edge = logical(img_edge(2:513,2:513));
end