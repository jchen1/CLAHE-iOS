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

// Reduces (adds up) the values in vec
int vredq_u8(uint8x16_t vec) {
    uint8x8_t high = vget_high_u8(vec), low = vget_low_u8(vec);
    
    uint16x4_t acc1_high = vpaddl_u8(high);
    uint16x4_t acc1_low = vpaddl_u8(low);
    
    uint32x2_t acc2_high = vpaddl_u16(acc1_high);
    uint32x2_t acc2_low = vpaddl_u16(acc1_low);
    
    uint64x1_t acc3_high = vpaddl_u32(acc2_high);
    uint64x1_t acc3_low = vpaddl_u32(acc2_low);
    
    return (int)(vget_lane_u64(acc3_high, 0) + vget_lane_u64(acc3_low, 0));
}

int get_excess(uint8_t* hist, int num_bins, int clip_limit) {
    uint8x16_t vec_limit = vdupq_n_u8(clip_limit);
    
    uint8x16_t acc = vdupq_n_u8(0);
    
    for (int i = 0; i < num_bins; i += 16) {
        uint8x16_t x = vld1q_u8(hist + i);
        uint8x16_t mask = vcgtq_u8(x, vec_limit);
        
        uint8x16_t x_masked = vandq_u8(x, mask);
        uint8x16_t clip_masked = vandq_u8(vec_limit, mask);
        
        uint8x16_t excess_vec = vsubq_u8(x_masked, clip_masked);
        acc = vaddq_u16(acc, excess_vec);
    }
    
    return vredq_u8(acc);
}

void clip_histogram(uint8_t* hist, int num_bins, int clip_limit) {
    int num_excess = 0, incr, upper;
    
    num_excess = get_excess(hist, num_bins, clip_limit);
    
    incr = num_excess / num_bins;
    upper = clip_limit - incr;
    
    uint8x16_t vec_limit = vdupq_n_u8(clip_limit);
    uint8x16_t vec_upper = vdupq_n_u8(upper);
    uint8x16_t vec_incr = vdupq_n_u8(incr);
    
    for (int i = 0; i < num_bins; i += 16) {
        uint8x16_t x = vld1q_u8(hist + i);
        
        // x_masked holds vals > upper and < clip_limit
        uint8x16_t mask_limit = vcltq_u8(x, vec_limit);
        uint8x16_t x_masked = vandq_u8(x, mask_limit);
        
        uint8x16_t mask_upper = vcgtq_u8(x_masked, vec_upper);
        x_masked = vandq_u8(x_masked, mask_upper);
        
        uint8x16_t upper_masked = vandq_u8(vec_upper, mask_upper);
        
        // num_excess -= hist[i] - upper;
        num_excess -= vredq_u8(vsubq_u8(x_masked, upper_masked));
        
        // now x_masked is all elements <= upper
        mask_upper = vcleq_u8(x, vec_upper);
        x_masked = vandq_u8(x, mask_upper);
        
        // num_excess -= incr for each hist[i] <= upper
        uint8x16_t incr_masked = vandq_u8(vec_incr, mask_upper);
        num_excess -= vredq_u8(incr_masked);
        
        // set hist[i] = min(hist[i] + incr, limit)
        x = vaddq_u8(x, vec_incr);
        x = vminq_u8(x, vec_limit);
        
        vst1q_u8(hist + i, x);
    }
    
    while (num_excess) {
        uint8_t* end = hist + num_bins, *cur = hist;
        
        while (num_excess && cur < end) {
            int step_size = num_bins / num_excess;
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

void map_histogram(uint8_t* hist, int num_bins, int min, int max, size_t num_pixels) {
    const int fScale = 256 / num_pixels;
    
    uint8x16_t vec_scale = vdupq_n_u8(fScale);
    uint8x16_t vec_zero = vdupq_n_u8(0);
    
    // this is parallelizable, but the parent function is already being parallelized
    for (int i = 0; i < num_bins; i += 16) {
        
        uint8x16_t x = vld1q_u8(hist + i);
        
        x = vaddq_u8(x, vextq_u8(vec_zero, x, 15));
        x = vaddq_u8(x, vextq_u8(vec_zero, x, 14));
        x = vaddq_u8(x, vextq_u8(vec_zero, x, 12));
        x = vaddq_u8(x, vextq_u8(vec_zero, x, 8));
        
        vst1q_u8(hist + i, x);
    }
    
    
    uint8x16_t x = vld1q_u8(hist);
    uint8x16_t add = vdupq_n_u8(vgetq_lane_u8(x, 15));
    
    vst1q_u8(hist, vmulq_u8(x, vec_scale));
    
    for (int i = 16; i < num_bins; i += 16) {
        x = vld1q_u8(hist + i);
        x = vaddq_u8(x, add);
        add = vdupq_n_u8(vgetq_lane_u8(x, 15));
        
        vst1q_u8(hist + i, vmulq_u8(x, vec_scale));
    }
}

void interpolate(cv::Mat in, uint8_t* histLU, uint8_t* histRU, uint8_t* histLB, uint8_t* histRB, int tile_size, int start_x, int start_y, cv::Mat out, uint8_t rshift, uint8_t log) {
    
    int xCoef, xInvCoef, yCoef, yInvCoef;
    
    auto in_ptr = in.ptr(), out_ptr = out.ptr();
    
    for (yCoef = 0, yInvCoef = tile_size; yCoef < tile_size; yCoef++, yInvCoef--) {
        for (xCoef = 0, xInvCoef = tile_size; xCoef < tile_size; xCoef++, xInvCoef--) {
            int x = start_x + xCoef, y = start_y + yCoef;
            if (y < out.rows && x < out.cols) {
                auto val = in_ptr[y*out.step + x] >> rshift;
                out_ptr[y*out.step + x] =
                ((yInvCoef * (xInvCoef*histLU[val] + xCoef*histRU[val])
                  + yCoef * (xInvCoef*histLB[val] + xCoef*histRB[val])) >> log);
            }
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
    
    int clip_limit = (int) (cl * tile_size * tile_size / num_bins);
    if (clip_limit < 1) clip_limit = 1;
    
    int nrX = in.cols / tile_size, nrY = in.rows / tile_size;
    
    uint8_t* hists = (uint8_t*)calloc(num_bins*nrX * nrY, sizeof(uint8_t));
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    // Parallelize by tiles to avoid cache thrashing
    dispatch_apply(nrY, queue, ^(size_t tileY) {
        for (int i = 0; i < tile_size; i++) {
            make_histograms(hists, in, rshift, tile_size * tileY + i, tile_size, num_bins);
        }
    });
    
    dispatch_apply(nrY*nrX, queue, ^(size_t tile) {
        auto hist = hists + num_bins*(tile);
        
        clip_histogram(hist, num_bins, clip_limit);
        map_histogram(hist, num_bins, 0, 255, tile_size * tile_size);
    });
    
//    for (int tile = 0; tile < nrY*nrX; tile++) {
//        auto hist = hists + num_bins*(tile);
//        
//        clip_histogram(hist, num_bins, clip_limit);
//        map_histogram(hist, num_bins, 0, 255, tile_size * tile_size);
//    };
    
    int norm = tile_size * tile_size;
    int log = 0;
    while (norm>>=1) log++;
    
    dispatch_apply((nrY+1)*(nrX+1), queue, ^(size_t num) {
        int tileX = num%(nrX+1), tileY = num/(nrX+1), startX = 0, startY = 0;
        int tileYU, tileYB, tileXL, tileXR;
        
        uint8_t *histLU, *histRU, *histLB, *histRB;
        
        tileYU = tileY > 0 ? tileY - 1 : 0;
        tileYB = tileY == nrY ? tileY - 1 : tileY;
        
        startY = tileY ? (tile_size >> 1) + (tileY - 1) * tile_size : 0;
        
        tileXL = tileX > 0 ? tileX - 1 : 0;
        tileXR = tileX == nrX ? tileX - 1 : tileX;
        
        startX = tileX ? (tile_size >> 1) + (tileX - 1) * tile_size : 0;
        
        histLU = hists + (num_bins * (tileYU * nrX + tileXL));
        histRU = hists + (num_bins * (tileYU * nrX + tileXR));
        histLB = hists + (num_bins * (tileYB * nrX + tileXL));
        histRB = hists + (num_bins * (tileYB * nrX + tileXR));
    
        interpolate(in, histLU, histRU, histLB, histRB, tile_size, startX, startY, out, rshift, log);
    });
    
    free(hists);
    
    return out;
}