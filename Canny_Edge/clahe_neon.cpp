//
//  clahe_neon.cpp
//  Canny_Edge
//
//  Created by Jeff Chen on 12/9/15.
//  Copyright © 2015 Jeff Chen. All rights reserved.
//

#include <stdio.h>
#include "clahe.hpp"
#include <math.h>
#include <dispatch/dispatch.h>

#include <vector>
#include <arm_neon.h>

void make_histograms(uint8_t* hists, cv::Mat in, uint8_t rshift, int row, int tile_size, int num_bins) {
    int tile = ((row / tile_size)*(in.cols / tile_size)) - 1;

    auto in_row = in.ptr<uint8_t>(row);
    for (int col = 0; col < in.cols; col++) {
        if (col % tile_size == 0) tile++;
        
        (hists + num_bins*tile)[in_row[col] >> rshift]++;
    }
}

void make_histogram(uint8_t* hist, cv::Mat in, uint8_t rshift) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            hist[in.at<uchar>(i, j) >> rshift]++;
        }
    }
}

void clip_histogram(uint8_t* hist, int num_bins, long clip_limit, bool strict) {
    
    long num_excess = 0, incr, upper;
    for (int i = 0; i < num_bins; i++) {
        long excess = hist[i] - clip_limit;
        if (excess > 0) num_excess += excess;
    }
    
    incr = num_excess / num_bins;
    upper = clip_limit - incr;
    
    if (strict) {
        for (int i = 0; i < num_bins; i++) {
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
            uint8_t* end = hist + num_bins, *cur = hist;
            
            while (num_excess && cur < end) {
                long step_size = num_bins / num_excess;
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
        for (int i = 0; i < num_bins; i++) {
            if (hist[i] > clip_limit) {
                hist[i] = clip_limit + incr;
            }
            else {
                hist[i] += incr;
            }
        }
    }
    
}

void sum_histogram(uint8_t* hist, uint8_t* out, int num_bins, int fScale) {

    uint8x16_t vec_scale = vdupq_n_u8(fScale);
    
    // this is parallelizable, but the parent function is already being parallelized
    for (int i = 0; i < num_bins; i += 16) {
        
        uint8x16_t x = vld1q_u8(hist + i);
        
        x = vaddq_u8(x, vextq_u8(x, x, 1));
        x = vaddq_u8(x, vextq_u8(x, x, 2));
        x = vaddq_u8(x, vextq_u8(x, x, 4));
        x = vaddq_u8(x, vextq_u8(x, x, 8));
        
        vst1q_u8(out + i, x);
    }
    
    uint8x16_t x = vld1q_u8(out);
    uint8x16_t add = vdupq_lane_u8(vget_low_u8(x), 7);
    
    vst1q_u8(out, vmulq_u8(x, vec_scale));
    
    for (int i = 16; i < num_bins; i += 16) {
        x = vld1q_u8(out + i);
        
        x = vaddq_u8(x, add);
        
        vst1q_u8(out + i, vmulq_u8(x, vec_scale));
        add = vdupq_lane_u8(vget_low_u8(x), 7);
    }
}

void map_histogram(uint8_t* hist, int num_bins, int min, int max, size_t num_pixels) {
//    const double fScale = 255. / num_pixels;
    const int fScale = 256 / num_pixels;
    
//    sum_histogram(hist, hist, num_bins, fScale);
    
    long accum = 0;
    
    for (int i = 0; i < num_bins; i++) {
        accum += hist[i];
        //        hist[i] = roundl((accum - cdf_min) / (fScale) * (max - min)) + min;
        //        hist[i] = roundl((accum - cdf_min) / fScale * 255);
        hist[i] = (uint8_t)(accum * fScale);
        
        if (hist[i] > 255) hist[i] = 255;
        else if (hist[i] < 0) hist[i] = 0;
    }
    
}

void interpolate(cv::Mat in, uint8_t* histLU, uint8_t* histRU, uint8_t* histLB, uint8_t* histRB, int tile_size, int start_x, int start_y, cv::Mat out, uint8_t rshift, uint8_t log) {
    
    int xCoef, xInvCoef, yCoef, yInvCoef;
    
    auto in_ptr = in.ptr(), out_ptr = out.ptr();
    
    for (yCoef = 0, yInvCoef = tile_size; yCoef < tile_size; yCoef++, yInvCoef--) {
        for (xCoef = 0, xInvCoef = tile_size; xCoef < tile_size; xCoef++, xInvCoef--) {
            int x = start_x + xCoef, y = start_y + yCoef;
//            if (y < out.rows && x < out.cols) {
                auto val = in_ptr[y*out.step + x] >> rshift;
                out_ptr[y*out.step + x] =
                ((yInvCoef * (xInvCoef*histLU[val] + xCoef*histRU[val])
                  + yCoef * (xInvCoef*histLB[val] + xCoef*histRB[val])) >> log);
//            }
        }
    }
    
}


cv::Mat clahe_neon(cv::Mat in, cv::Mat mirrored, int tile_size, float cl, int num_bins) {
    if (in.channels() != 1) {
        printf("must be grayscale\n");
        return in;
    }
    
    if (tile_size & (tile_size - 1)) {
        printf("tile size must be power of 2\n");
        return in;
    }
    
    if (num_bins > 256 || num_bins & (num_bins-1)) {
        printf("bad num_bins\n");
        return in;
    }
    
    int rshift = 0, tmp = 512;
    while ((tmp>>=1) != num_bins) rshift++;
    
    cv::Mat out(in.rows, in.cols, in.type());
    
    long clip_limit = (long) (cl * tile_size * tile_size / num_bins);
    if (clip_limit < 1) clip_limit = 1;
    
    int nrX = in.cols / tile_size, nrY = in.rows / tile_size;
    
    uint8_t* hists = (uint8_t*)calloc(num_bins*nrX * nrY, sizeof(uint8_t));
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    // Don't parallelize this b/c of cache thrashing
    for (int i = 0; i < in.rows; i++) {
        make_histograms(hists, in, rshift, i, tile_size, num_bins);
    }
    
    dispatch_apply(nrY*nrX, queue, ^(size_t tile) {
        auto hist = hists + num_bins*(tile);
        auto i = tile/nrX;
        auto j = tile%nrX;
        
        cv::Mat roi = mirrored(cv::Range((int)(i * tile_size + tile_size/2), (int)(i * tile_size + 3*tile_size/2)), cv::Range((int)(j * tile_size + tile_size/2), (int)(j * tile_size + 3*tile_size/2)));
        
        clip_histogram(hist, num_bins, clip_limit, true);
        map_histogram(hist, num_bins, 0, 255, tile_size * tile_size);
    });
    
    int norm = tile_size * tile_size;
    int log = 0;
    while (norm>>=1) log++;
    
    dispatch_apply((nrY+1)*(nrX+1), queue, ^(size_t num) {
        int tileX = num%(nrX+1), tileY = num/(nrX+1), startX = 0, startY = 0;
        int tileYU, tileYB, tileXL, tileXR;
        
        uint8_t *histLU, *histRU, *histLB, *histRB;

        if (tileY == 0) {
            tileYU = 0; tileYB = 0;
            startY = 0;
        }
        else {
            if (tileY == nrY) {
                tileYU = nrY - 1;
                tileYB = tileYU;
            }
            else {
                tileYU = tileY - 1;
                tileYB = tileY;
            }
            startY = (tile_size>>1) + (tileY-1)*tile_size;
        }
        
        if (tileX == 0) {
            tileXL = 0; tileXR = 0;
            startX = 0;
        }
        else {
            if (tileX == nrX) {
                tileXL = nrX - 1;
                tileXR = tileXL;
            }
            else {
                tileXL = tileX - 1;
                tileXR = tileX;
            }
            startX = (tile_size>>1) + (tileX-1)*tile_size;
        }
        
        histLU = hists + (num_bins * (tileYU * nrX + tileXL));
        histRU = hists + (num_bins * (tileYU * nrX + tileXR));
        histLB = hists + (num_bins * (tileYB * nrX + tileXL));
        histRB = hists + (num_bins * (tileYB * nrX + tileXR));
    
        interpolate(in, histLU, histRU, histLB, histRB, tile_size, startX, startY, out, rshift, log);
    });
    
    free(hists);
    
    return out;
}