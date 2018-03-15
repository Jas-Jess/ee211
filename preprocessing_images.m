clear
close all

%% Load image, change to double
srcFiles = dir('hgr2b_original_images/*_A_*.jpg');  % the folder in which ur images exists
for k = 1 : length(srcFiles)
    tic
    filename = strcat('hgr2b_original_images/',srcFiles(k).name);
    srcFiles(k).name
%     if('0_A_hgr2B_id12_1.jpg' == srcFiles(k).name)
%         srcFiles(k).name
%     end
    image = imread(filename);
    image = imresize(image, [1024 NaN]);
    img = rgb2gray(im2double(image));
    [m,n] = size(img);
%     figure(1), imshow(image);

    %% Sobel Filter
    sobelImg = edge(img,'Sobel');
%     figure(2), imshow(sobelImg);

    %% Diliate image
    diImg = bwmorph(sobelImg,'dilate');       %THIS ONE
    diImg = bwmorph(diImg,'skel');
    diImg = bwmorph(diImg,'branchpoints');
    diImg = bwmorph(diImg,'open');
    diImg = bwmorph(diImg,'majority');
    diImg = bwmorph(diImg,'thicken');
    diImg = bwmorph(diImg,'thicken');
    diImg = bwmorph(diImg,'thicken');
%     figure(3), imshow(diImg);

    %% Fill holes
    filledImg = imfill(diImg, 'holes');   
    filledImg = bwmorph(filledImg,'thicken');
%     figure(4), imshow(filledImg);

    %% Smooth with mask
    mask = strel('rectangle',[3,6]);
    maskedImg = imdilate(filledImg,mask);
    maskedImg = imdilate(maskedImg,mask);
    maskedImg = bwmorph(maskedImg,'majority');
%     figure(5), imshow(maskedImg);

    %% Check if we need bounding box
    if(sum(sum(maskedImg)) > 15000)
%         figure(5), imshow(maskedImg);
        maskedImg = bwmorph(maskedImg,'dilate');
        maskedImg = bwmorph(maskedImg,'dilate');
        maskedImg = bwmorph(maskedImg,'open');
%         figure(6), imshow(maskedImg);
        %% bounding box
        boxes = regionprops(bwareafilt(maskedImg,1),'BoundingBox'); 
        boxIdx = boxes(1).BoundingBox;

        %% removing rest from hand
        if(((boxIdx(2)+boxIdx(4)) > (m/4*3)) || (boxIdx(1)+boxIdx(3)) > (n/4*3))    %should add condition if near the top of left edge2
            theta = m*n/7000;
        else
            theta = 0;
        end
        maskedImg(ceil(boxIdx(2)-theta):ceil(boxIdx(2)+boxIdx(4)+theta),ceil(boxIdx(1)-theta):ceil(boxIdx(1)+boxIdx(3)+theta)) = 0;
%         figure(7), imshow(maskedImg);

    end

    %% segment image, find section with largest value
    splitImage=mat2tiles(maskedImg,ceil(size(maskedImg)/4));
    %find largest value cell
    maxSum = 0;
    for i = 1:4
        for j = 1:4
            currTile = cell2mat(splitImage(i,j));
            currSum = sum(sum(currTile(:,:)));

            if (currSum > maxSum)
                maxIndex = [i,j];
                maxSum = currSum;
            end
        end
    end

    %set minimum crop coordinates
    if(m > n) %vertical image
        mMin = floor(m/4*(maxIndex(1)-1) - (m/7));
        nMin = floor(n/4*(maxIndex(2)-1) - (n/7));
        uint8(mMin);
        uint8(nMin);
    else %horizontal image
        mMin = floor(m/4)*(maxIndex(1)-1) - (m/7);
        nMin = floor(n/4)*(maxIndex(2)-1) - (n/7);
        uint8(mMin);
        uint8(nMin);
    end

    width = ceil(n/1.8);
    height = ceil(m/1.8);

    %% Crop
    newImg = imcrop(image,[nMin mMin,width,height]);
    newImg = imresize(newImg, [500 500]);
%     figure(7), imshow(newImg);

    %% Save images into new folder
    fullFileName = fullfile('cropped_hgr2b_original/', srcFiles(k).name);
    imwrite(newImg, fullFileName);
    
    toc
end
