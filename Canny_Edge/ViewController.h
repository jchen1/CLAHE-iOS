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
    uint64_t prevTime;
}

@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (nonatomic, strong) IBOutlet UIImageView* imageView;

@end

