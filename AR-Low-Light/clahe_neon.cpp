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

// Given the array of histograms and the starting row of the image,
// properly update the histograms with the input image's row.
void make_histograms(uint16_t* hists, cv::Mat in, uint8_t rshift, int row, int tile_size, int num_bins) {
    int tile = ((row / tile_size)*(in.cols / tile_size)) - 1;
    
    auto in_row = in.ptr<uint8_t>(row);
    for (int col = 0; col < in.cols; col++) {
        if (col % tile_size == 0) tile++;
        
        (hists + num_bins*tile)[in_row[col] >> rshift]++;
    }
}

// Reduces (adds up) the 16 values in vec in 5 instructions (assuming
// instruction-level parallelism)
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

// Adds the 8 values in vec using 4 instructions (assuming instruction-level
// parallelism
int vredq_u16(uint16x8_t vec) {
    uint16x4_t high = vget_high_u16(vec), low = vget_low_u16(vec);
    
    uint32x2_t acc1_high = vpaddl_u16(high);
    uint32x2_t acc1_low = vpaddl_u16(low);
    
    uint64x1_t acc2_high = vpaddl_u16(acc1_high);
    uint64x1_t acc2_low = vpaddl_u16(acc1_low);
    
    return (int)(vget_lane_u64(acc2_high, 0) + vget_lane_u64(acc2_low, 0));
}

// Get the area of the histogram above clip_limit
int get_excess(uint16_t* hist, int num_bins, int clip_limit) {
    auto vec_limit = vdupq_n_u16(clip_limit);
    
    auto acc = vdupq_n_u16(0);
    
    for (int i = 0; i < num_bins; i += 8) {
        auto x = vld1q_u16(hist + i);
        auto mask = vcgtq_u16(x, vec_limit);
        
        auto x_masked = vandq_u16(x, mask);
        auto clip_masked = vandq_u16(vec_limit, mask);
        
        auto excess_vec = vsubq_u16(x_masked, clip_masked);
        acc = vaddq_u16(acc, excess_vec);
    }
    
    return vredq_u8(acc);
}

// Clips the histogram, ensuring that no bin is greater than clip_limit.
// The excess data is then evenly distributed amongst the remainder
// of the histogram.

void clip_histogram(uint16_t* hist, int num_bins, int clip_limit) {
    int num_excess = 0, incr, upper;
    
    num_excess = get_excess(hist, num_bins, clip_limit);
    
    incr = num_excess / num_bins;
    upper = clip_limit - incr;
    
    auto vec_limit = vdupq_n_u16(clip_limit);
    auto vec_upper = vdupq_n_u16(upper);
    auto vec_incr = vdupq_n_u16(incr);
    
    // Perform the initial clipping
    for (int i = 0; i < num_bins; i += 8) {
        auto x = vld1q_u16(hist + i);
        
        // x_masked holds vals > upper and < clip_limit
        auto mask_limit = vcltq_u16(x, vec_limit);
        auto x_masked = vandq_u16(x, mask_limit);
        
        auto mask_upper = vcgtq_u16(x_masked, vec_upper);
        x_masked = vandq_u16(x_masked, mask_upper);
        
        auto upper_masked = vandq_u16(vec_upper, mask_upper);
        
        // num_excess -= hist[i] - upper;
        num_excess -= vredq_u16(vsubq_u16(x_masked, upper_masked));
        
        // now x_masked is all elements <= upper
        mask_upper = vcleq_u16(x, vec_upper);
        x_masked = vandq_u16(x, mask_upper);
        
        // num_excess -= incr for each hist[i] <= upper
        auto incr_masked = vandq_u16(vec_incr, mask_upper);
        num_excess -= vredq_u16(incr_masked);
        
        // set hist[i] = min(hist[i] + incr, limit)
        x = vaddq_u16(x, vec_incr);
        x = vminq_u16(x, vec_limit);
        
        vst1q_u16(hist + i, x);
    }
    
    // Distribute the remaining excess equally across bins, ensuring that
    // no bin ever exceeds clip_limit
    while (num_excess) {
        uint16_t* end = hist + num_bins, *cur = hist;
        
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

// Converts hist into a CDF, effectively mapping input intensities to output intensities
void map_histogram(uint16_t* hist, int num_bins, size_t num_pixels) {
    const int fScale = 256 / num_pixels;
    
    auto vec_scale = vdupq_n_u16(fScale);
    auto vec_zero = vdupq_n_u16(0);
    
    // this is parallelizable, but the parent function is already being parallelized
    for (int i = 0; i < num_bins; i += 8) {
        
        auto x = vld1q_u16(hist + i);
        
        x = vaddq_u16(x, vextq_u16(vec_zero, x, 7));
        x = vaddq_u16(x, vextq_u16(vec_zero, x, 6));
        x = vaddq_u16(x, vextq_u16(vec_zero, x, 4));
        
        vst1q_u16(hist + i, x);
    }
    
    
    auto x = vld1q_u16(hist);
    auto max_vec = vdupq_n_u16(255);
    auto add = vdupq_n_u16(vgetq_lane_u16(x, 7));
    
    vst1q_u16(hist, vminq_u16(vmulq_u16(x, vec_scale), max_vec));
    
    for (int i = 8; i < num_bins; i += 8) {
        x = vld1q_u16(hist + i);
        x = vaddq_u16(x, add);
        add = vdupq_n_u16(vgetq_lane_u16(x, 7));
        
        vst1q_u16(hist + i, vminq_u16(vmulq_u16(x, vec_scale), max_vec));
    }
}

// For a given tile and three surrounding it, interpolate the tiles' histograms
// to obtain a final intensity value for each pixel in the tile
void interpolate(cv::Mat in, uint16_t* histLU, uint16_t* histRU, uint16_t* histLB, uint16_t* histRB, int tile_size, int start_x, int start_y, cv::Mat out, uint8_t rshift, uint8_t log) {
    
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

// Performs CLAHE on the input image using GCD and NEON.
//
// tile_size should be between 4 and 16, usually 8.
//
// clip_limit affects how much brighter a pixel can get (a high clip_limit will
// reduce the total contrast of the image because dark pixels will be transformed
// into almost-white pixels).
//
// num_bins should be a power of 2 that is at most 256. It is the size of each
// histogram. To improve performance but reduce quality, shrink num_bins to 128 or 64.
// Anything below 64 is largely unusable.

cv::Mat clahe_neon(cv::Mat in, int tile_size, int clip_limit, int num_bins, bool should_interpolate) {
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
    
    cv::Mat out(in.rows, in.cols, in.type());

    int nrX = in.cols / tile_size, nrY = in.rows / tile_size;
    int rshift = 0, tmp = 512;
    while ((tmp>>=1) != num_bins) rshift++;
    
    int norm = tile_size * tile_size;
    int log = 0;
    while (norm>>=1) log++;
    
    uint16_t* hists = (uint16_t*)calloc(num_bins*nrX * nrY, sizeof(uint16_t));
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    // Each task handles a single row of tiles to avoid cache thrashing
    dispatch_apply(nrY, queue, ^(size_t tileY) {
        for (int i = 0; i < tile_size; i++) {
            make_histograms(hists, in, rshift, tile_size * tileY + i, tile_size, num_bins);
        }
    });
    
    // Each task clips, then maps the histogram
    dispatch_apply(nrY*nrX, queue, ^(size_t tile) {
        auto hist = hists + num_bins*(tile);
        
        clip_histogram(hist, num_bins, clip_limit);
        map_histogram(hist, num_bins, tile_size * tile_size);
    });
    
    if (should_interpolate) {
        // Interpolate, parallelizing across tiles.
        dispatch_apply((nrY+1)*(nrX+1), queue, ^(size_t num) {
            
            // Choose the four closest tiles and grab their histograms.
            // If the tile is on an edge or corner, choose fewer tiles.
            int tileX = num%(nrX+1), tileY = num/(nrX+1), startX = 0, startY = 0;
            int tileYU, tileYB, tileXL, tileXR;
            
            uint16_t *histLU, *histRU, *histLB, *histRB;
            
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
    }
    
    else {
        dispatch_apply((nrY)*(nrX), queue, ^(size_t num) {
            int tileX = num%(nrX), tileY = num/nrX;
            
            int startY = tileY ? (tile_size >> 1) + (tileY - 1) * tile_size : 0;
            int startX = tileX ? (tile_size >> 1) + (tileX - 1) * tile_size : 0;
            
            uint16_t* hist = hists + (num_bins * (tileY * nrX + tileX));
            
            interpolate(in, hist, hist, hist, hist, tile_size, startX, startY, out, rshift, log);
        });
    }
    

    free(hists);
    
    return out;
}