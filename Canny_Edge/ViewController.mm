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
    AVCaptureSessionPreset1280x720;
    videoCamera.defaultAVCaptureVideoOrientation =
    AVCaptureVideoOrientationPortrait;
    videoCamera.defaultFPS = 30;
    
    // Read in the image
//    UIImage *image = [UIImage imageNamed:@"720.jpg"];
//    if(image == nil) cout << "Cannot read in the file prince_book.jpg!!" << endl;
    
    // Setup the display
    // Setup the your imageView_ view, so it takes up the entire App screen......
//    imageView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
//    // Important: add OpenCV_View as a subview
//    [self.view addSubview:imageView];
//    
//    // Ensure aspect ratio looks correct
//    imageView.contentMode = UIViewContentModeScaleAspectFit;
//    
    // Another way to convert between cvMat and UIImage (using member functions)
    
//    cv::Mat cvImage = [self cvMatFromUIImage:image];
//    cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
//    cv::Mat display_im; cv::cvtColor(gray,display_im,CV_GRAY2BGR); // Get the display image
//    
////    cv::Mat le = clahe_naive(gray, 8, 2.0);
//    
//    struct timeval time;
//    gettimeofday(&time, NULL);
//    long start = (time.tv_sec * 1000) + (time.tv_usec / 1000);
//    
//    int num_imgs = 50;
//    
//    int tile_size = 8;
//    
//    cv::Mat mirrored;
//    cv::copyMakeBorder(gray, mirrored, tile_size, tile_size, tile_size, tile_size, cv::BORDER_REFLECT);
//    
//    cv::Mat le;
//    for (int i = 0; i < num_imgs; i++){
////        printf("%d\n", i);
////        le = clahe_interp(gray, mirrored, tile_size, 3.0);
//        le = clahe_neon(gray, mirrored, tile_size, 3.0, 256);
////        le = clahe_naive(gray, 8, 3.0);
//    }
//    
//    gettimeofday(&time, NULL);
//    long finish = (time.tv_sec * 1000) + (time.tv_usec / 1000);
//    
//    printf("took %ld ms to finish, %f ms per img, %f FPS\n", finish - start, (finish - start)/(num_imgs*1.), 1000./ ((finish - start)/(num_imgs*1.)));
//    display_im = le;
////    display_im = gray;
//
//    cv::cvtColor(display_im, display_im, CV_GRAY2RGBA);
    
    
    // Finally setup the view to display
    [videoCamera start];
//    imageView.image = [self UIImageFromCVMat:display_im];
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
    
    // Apply filter
    cv::Mat gray; cv::cvtColor(image, gray, CV_RGBA2GRAY); // Convert to grayscale
//    cv::Mat display_im; cv::cvtColor(gray,display_im,CV_GRAY2BGR); // Get the display image
    
    //    cv::Mat le = clahe_naive(gray, 8, 2.0);
    
    int num_imgs = 50;
    
    int tile_size = 8;
    
    cv::Mat mirrored;
    cv::copyMakeBorder(gray, mirrored, tile_size/2, tile_size/2, tile_size/2, tile_size/2, cv::BORDER_REFLECT);
    cv::Mat clahe = clahe_neon(gray, mirrored, tile_size, 5.0, 256);
    
    if (isNeedRotation)
        clahe = clahe.t();
    
    // Add fps label to the frame
    uint64_t currTime = mach_absolute_time();
    double timeInSeconds = machTimeToSecs(currTime - prevTime);
    prevTime = currTime;
    double fps = 1.0 / timeInSeconds;
    NSString* fpsString =
    [NSString stringWithFormat:@"FPS = %3.2f", fps];
    cv::putText(clahe, [fpsString UTF8String],
                cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cv::Scalar::all(255));
    
    clahe.copyTo(image);
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

//---------------------------------------------------------------------------------------------------------------------
// You should not have to touch these functions below to complete the assignment!!!!
//---------------------------------------------------------------------------------------------------------------------

// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

@end
