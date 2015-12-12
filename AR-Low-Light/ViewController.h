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
    BOOL isHE;
    bool color;
    int tile_size;
    int clip_limit;
    int num_bins;
    bool should_interpolate;
    uint64_t prevTime;
}

@property (nonatomic, strong) IBOutlet UIToolbar* camToolbar;
@property (nonatomic, strong) IBOutlet UIToolbar* claheToolbar;

@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (nonatomic, strong) IBOutlet UIImageView* imageView;

@property (nonatomic, weak) IBOutlet
UIBarButtonItem* toggleCaptureButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* toggleCLAHEButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* toggleColorButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* toggleInterpolateButton;
@property (nonatomic, weak) IBOutlet
UIBarButtonItem* toggleHEButton;

-(IBAction)toggleCaptureButtonPressed:(id)sender;
-(IBAction)toggleInterpolateButtonPressed:(id)sender;
-(IBAction)toggleColorButtonPressed:(id)sender;
-(IBAction)toggleCLAHEButtonPressed:(id)sender;
-(IBAction)toggleHEButtonPressed:(id)sender;




@end

