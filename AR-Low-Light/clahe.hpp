//
//  clahe.hpp
//  Canny_Edge
//
//  Created by Jeff Chen on 12/8/15.
//  Copyright © 2015 Jeff Chen. All rights reserved.
//

#ifndef clahe_hpp
#define clahe_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp> // Includes the opencv library

// CLAHE with tiling, highly optimized with GCD and NEON
cv::Mat clahe_neon(cv::Mat in, int tile_size, int cl, int num_bins);

// CLAHE with tiling and interpolation
cv::Mat clahe_interp(cv::Mat in, cv::Mat mirrored, int tile_size, float cl);

// Naive CLAHE implementation: slow
cv::Mat clahe_naive(cv::Mat in, int tile_size, float cl);

// Naive simple HE implementation
cv::Mat he_naive(cv::Mat in);

#endif /* clahe_hpp */
