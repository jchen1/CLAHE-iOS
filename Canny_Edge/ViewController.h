//
//  ViewController.h
//  Canny_Edge
//
//  Created by Jeff Chen on 10/14/15.
//  Copyright © 2015 Jeff Chen. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <opencv2/highgui/ios.h> 

@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    CvVideoCamera* videoCamera;
    BOOL isCapturing;
    BOOL isCLAHE;
    uint64_t prevTime;
}

@property (nonatomic, strong) IBOutlet UIToolbar* camToolbar;
@property (nonatomic, strong) IBOutlet UIToolbar* claheToolbar;

@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (nonatomic, strong) IBOutlet UIImageView* imageView;

@property (nonatomic, weak) IBOutlet
UIBarButtonItem* startCaptureButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* stopCaptureButton;

-(IBAction)startCaptureButtonPressed:(id)sender;
-(IBAction)stopCaptureButtonPressed:(id)sender;

@property (nonatomic, weak) IBOutlet
UIBarButtonItem* startCLAHEButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* stopCLAHEButton;

-(IBAction)startCLAHEButtonPressed:(id)sender;
-(IBAction)stopCLAHEButtonPressed:(id)sender;

@end

