//
//  ViewController.m
//  Estimate_PlanarWarps
//
//  Created by Simon Lucey on 9/21/15.
//  Copyright (c) 2015 CMU_16432. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#include <opencv2/opencv.hpp> // Includes the opencv library
#include <stdlib.h> // Include the standard library
#include "opencv2/nonfree/nonfree.hpp"

#include "clahe.hpp"

#import <mach/mach_time.h>

#endif

using namespace std;

@interface ViewController () {

}
@end

@implementation ViewController

@synthesize imageView;
@synthesize videoCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Initialize camera
    videoCamera = [[CvVideoCamera alloc]
                   initWithParentView:imageView];
    videoCamera.delegate = self;
    videoCamera.defaultAVCaptureDevicePosition =
    AVCaptureDevicePositionBack;
    videoCamera.defaultAVCaptureSessionPreset =
//    AVCaptureSessionPreset1920x1080;
    AVCaptureSessionPreset1280x720;
    videoCamera.defaultAVCaptureVideoOrientation =
    AVCaptureVideoOrientationPortrait;
    videoCamera.defaultFPS = 30;
    
    // Set up default settings
    isCLAHE = YES;
    isCapturing = NO;
    
    color = true;
    tile_size = 8;
    clip_limit = 2;
    num_bins = 256;
    should_interpolate = true;
}

//TODO: may be remove this code
static double machTimeToSecs(uint64_t time)
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    return (double)time * (double)timebase.numer /
    (double)timebase.denom / 1e9;
}

- (void)processImage:(cv::Mat&)image
{
    cv::Mat inputFrame = image;
    
    BOOL isNeedRotation = false;//image.size() != params.frameSize;
    
    if (isNeedRotation)
        inputFrame = image.t();
    
    cv::Mat clahe;
    
    if (isCLAHE) {
        // Apply filter
        
        if (color) {
            clahe = cv::Mat(image.rows, image.cols, CV_8UC3);
            
            cv::Mat hsv(image.rows, image.cols, CV_8UC3); cv::cvtColor(image, hsv, CV_BGRA2RGB); cv::cvtColor(hsv, hsv, CV_RGB2HLS);
            cv::Mat v(image.rows, image.cols, CV_8UC1); cv::extractChannel(hsv, v, 1);
            
            v = clahe_neon(v, tile_size, clip_limit, num_bins, should_interpolate);
            
            cv::Mat out[] = {hsv, v};
            int from_to[] = { 0,0, 3,1, 2,2 };
            cv::mixChannels(out, 2, &clahe, 1, from_to, 3);
            
            cv::cvtColor(clahe, clahe, CV_HLS2RGB);
        }
        else {
            cv::Mat gray; cv::cvtColor(image, gray, CV_RGBA2GRAY);
            clahe = clahe_neon(gray, tile_size, clip_limit, num_bins, should_interpolate);
        }
        
    }
    else if (isHE) {
        if (color) {
            clahe = cv::Mat(image.rows, image.cols, CV_8UC3);
            
            cv::Mat hsv(image.rows, image.cols, CV_8UC3); cv::cvtColor(image, hsv, CV_BGRA2RGB); cv::cvtColor(hsv, hsv, CV_RGB2HLS);
            cv::Mat v(image.rows, image.cols, CV_8UC1); cv::extractChannel(hsv, v, 1);
            
            v = he_naive(v);
            
            cv::Mat out[] = {hsv, v};
            int from_to[] = { 0,0, 3,1, 2,2 };
            cv::mixChannels(out, 2, &clahe, 1, from_to, 3);
            
            cv::cvtColor(clahe, clahe, CV_HLS2RGB);
        }
        else {
            cv::Mat gray; cv::cvtColor(image, gray, CV_RGBA2GRAY);
            clahe = he_naive(gray);
        }
    }
    
    else {
        if (!color) {
            cv::cvtColor(image, clahe, CV_RGBA2GRAY);
        }
        else {
            clahe = image;
        }
    }
    
    if (isHE || isCLAHE) {
        // Add fps label to the frame
        cv::rectangle(clahe, cv::Point(25, 35), cv::Point(255, 90), cv::Scalar::all(0), CV_FILLED);
        
        uint64_t currTime = mach_absolute_time();
        double timeInSeconds = machTimeToSecs(currTime - prevTime);
        prevTime = currTime;
        double fps = 1.0 / timeInSeconds;
        NSString* fpsString =
        [NSString stringWithFormat:@"FPS = %3.2f", fps];
        cv::putText(clahe, [fpsString UTF8String],
                    cv::Point(30, 70), cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1.5, cv::Scalar::all(255));
    }

    
    clahe.copyTo(image);
}

-(IBAction)toggleCaptureButtonPressed:(id)sender
{
    isCapturing = !isCapturing;
    (isCapturing) ? [videoCamera start] : [videoCamera stop];
}

-(IBAction)toggleColorButtonPressed:(id)sender
{
    color = !color;
}

-(IBAction)toggleInterpolateButtonPressed:(id)sender
{
    should_interpolate = !should_interpolate;
}

-(IBAction)toggleCLAHEButtonPressed:(id)sender
{
    isCLAHE = !isCLAHE;
    if (isCLAHE) {
        isHE = NO;
    }
}

-(IBAction)toggleHEButtonPressed:(id)sender
{
    isHE = !isHE;
    if (isHE) {
        isCLAHE = NO;
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
