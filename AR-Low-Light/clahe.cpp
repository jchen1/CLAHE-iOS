//
//  clahe.cpp
//  Canny_Edge
//
//  Created by Jeff Chen on 12/8/15.
//  Copyright © 2015 Jeff Chen. All rights reserved.
//

#include "clahe.hpp"
#include <math.h>

#include <vector>

void make_histogram(long* hist, cv::Mat in) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            hist[in.at<uchar>(i, j)]++;
        }
    }
}

void clip_histogram(long* hist, long clip_limit, bool strict) {
    
    long num_excess = 0, incr, upper;
    for (int i = 0; i < 256; i++) {
        long excess = hist[i] - clip_limit;
        if (excess > 0) num_excess += excess;
    }
    
    incr = num_excess / 256;
    upper = clip_limit - incr;
    
    if (strict) {
        for (int i = 0; i < 256; i++) {
            if (hist[i] > clip_limit) hist[i] = clip_limit;
            else {
                if (hist[i] > upper) {
                    num_excess -= hist[i] - upper;
                    hist[i] = clip_limit;
                }
                else {
                    num_excess -= incr;
                    hist[i] += incr;
                }
            }
        }
        
        while (num_excess) {
            long* end = hist + 256, *cur = hist;
            
            while (num_excess && cur < end) {
                long step_size = 256 / num_excess;
                if (step_size < 1) step_size = 1;
                for (auto bin = cur; bin < end && num_excess; bin += step_size) {
                    if (*bin < clip_limit) {
                        (*bin)++;
                        num_excess--;
                    }
                }
                cur++;
            }
            
        }
    }
    
    else {
        for (int i = 0; i < 256; i++) {
            if (hist[i] > clip_limit) {
                hist[i] = clip_limit + incr;
            }
            else {
                hist[i] += incr;
            }
        }
    }

}

void map_histogram(long* hist, int min, int max, size_t num_pixels) {
    const double fScale = 255. / num_pixels;
    
    long accum = 0;
    
    for (int i = 0; i < 256; i++) {
        accum += hist[i];
        hist[i] = (unsigned long)(accum * fScale);
        
        if (hist[i] > 255) hist[i] = 255;
        else if (hist[i] < 0) hist[i] = 0;
    }
    
}

void interpolate(cv::Mat in, long* histLU, long* histRU, long* histLB, long* histRB, int tile_size, int start_x, int start_y, cv::Mat out) {
    
    int norm = tile_size * tile_size;
    int xCoef, xInvCoef, yCoef, yInvCoef;
    
    for (yCoef = 0, yInvCoef = tile_size; yCoef < tile_size; yCoef++, yInvCoef--) {
        for (xCoef = 0, xInvCoef = tile_size; xCoef < tile_size; xCoef++, xInvCoef--) {
            if (start_y + yCoef < out.rows && start_x + xCoef < out.cols) {
                uchar val = in.at<uchar>(start_y + yCoef, start_x + xCoef);
                
                out.at<uchar>(start_y + yCoef, start_x + xCoef) =
                (uchar)((yInvCoef * (xInvCoef*histLU[val] + xCoef*histRU[val])
                         + yCoef * (xInvCoef*histLB[val] + xCoef*histRB[val])) / norm);
            }

        }
    }
    
}

cv::Mat clahe_interp(cv::Mat in, cv::Mat mirrored, int tile_size, float cl) {
    if (in.channels() != 1) {
        printf("must be grayscale\n");
        return in;
    }
    
    if (tile_size % 2) {
        printf("tile size must be even\n");
        return in;
    }
    
    cv::Mat out(in.rows, in.cols, in.type());
    
    long clip_limit = (long) (cl * tile_size * tile_size / 256);
    if (clip_limit < 1) clip_limit = 1;
    
    int nrX = in.cols / tile_size, nrY = in.rows / tile_size;
    
    long* hists = (long*)calloc(256*nrX * nrY, sizeof(long));
    
    for (int i = 0; i < nrY; i++) {
        for (int j = 0; j < nrX; j++) {
            long* hist = hists + 256*(i*nrX + j);

            cv::Mat roi = mirrored(cv::Range(i * tile_size + tile_size/2, i * tile_size + 3*tile_size/2), cv::Range(j * tile_size + tile_size/2, j * tile_size + 3*tile_size/2));
            make_histogram(hist, roi);
            clip_histogram(hist, clip_limit, true);
            map_histogram(hist, 0, 255, tile_size * tile_size);
        }
    }
    
    int tileX = 0, tileY = 0, startX = 0, startY = 0;
    
    int tileYU, tileYB, tileXL, tileXR;
    
    long *histLU, *histRU, *histLB, *histRB;
    for (tileY = 0; tileY <= nrY; tileY++) {
        if (tileY == 0) {
            tileYU = 0; tileYB = 0;
        }
        else if (tileY == nrY) {
            tileYU = nrY - 1;
            tileYB = tileYU;
        }
        else {
            tileYU = tileY - 1;
            tileYB = tileY;
        }
        for (tileX = 0, startX = 0; tileX <= nrX; tileX++) {
            if (tileX == 0) {
                tileXL = 0; tileXR = 0;
            }
            else if (tileX == nrX) {
                tileXL = nrX - 1;
                tileXR = tileXL;
            }
            else {
                tileXL = tileX - 1;
                tileXR = tileX;
            }
            
            histLU = hists + (256 * (tileYU * nrX + tileXL));
            histRU = hists + (256 * (tileYU * nrX + tileXR));
            histLB = hists + (256 * (tileYB * nrX + tileXL));
            histRB = hists + (256 * (tileYB * nrX + tileXR));
        
            interpolate(in, histLU, histRU, histLB, histRB, tile_size, startX, startY, out);
            
            if (tileX == 0) startX += tile_size/2;
            else startX += tile_size;
        }
        
        if (tileY == 0) startY += tile_size/2;
        else startY += tile_size;
        
    }
    
    free(hists);
    
    return out;
}


cv::Mat clahe_naive(cv::Mat in, int tile_size, float cl) {
    if (in.channels() != 1) {
        printf("must be grayscale\n");
        return in;
    }
    
    if (tile_size % 2) {
        printf("tile size must be even\n");
        return in;
    }
    
    cv::Mat mirrored;
    cv::copyMakeBorder(in, mirrored, tile_size, tile_size, tile_size, tile_size, cv::BORDER_REFLECT);
    
    cv::Mat out(in.rows, in.cols, in.type());
    
    long clip_limit = (long) (cl * tile_size * tile_size / 256);
    if (clip_limit < 1) clip_limit = 1;
    
//    clip_limit = 40;
    
//    printf("clip limit: %d\n", clip_limit);
    
    long* hist = (long*)calloc(256, sizeof(long));
    
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            cv::Mat roi = mirrored(cv::Range(i + tile_size/2, i + 3*tile_size/2), cv::Range(j + tile_size/2, j + 3*tile_size/2));
            memset(hist, 0, 256 * sizeof(long));
            make_histogram(hist, roi);
            auto min = 255, max = 0;
            
//            for (int i = 0; i < 256; i++) {
//                if (hist[i] > 0) {
//                    if (min > i) {
//                        min = i;
//                    }
//                    if (max < i) {
//                        max = i;
//                    }
//                }
//            }
            
            clip_histogram(hist, clip_limit, false);
            map_histogram(hist, min, max, tile_size * tile_size);
            
//            auto val = hist[in.at<uchar>(i, j)];
//            printf("%ld\n", val);
            
            out.at<uchar>(i, j) = hist[in.at<uchar>(i, j)];
            
        }
    }
    
    free(hist);
    
    
    return out;
}

cv::Mat he_naive(cv::Mat in) {
    if (in.channels() != 1) {
        printf("must be grayscale\n");
        return in;
    }
    
    long* hist = (long*) calloc(256, sizeof(long));
    
    make_histogram(hist, in);
    
    map_histogram(hist, 255, 0, in.rows * in.cols);
    
    cv::Mat out(in.rows, in.cols, in.type());
    
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            uchar val = hist[in.at<uchar>(i, j)];
            out.at<uchar>(i, j) = val;
        }
    }
    
    free(hist);
    
    return out;
}

