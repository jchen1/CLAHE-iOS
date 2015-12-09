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

cv::Mat clahe_interp(cv::Mat in, int tile_size, float cl);
cv::Mat clahe_naive(cv::Mat in, int tile_size, float cl);

cv::Mat he_naive(cv::Mat in);

#endif /* clahe_hpp */
