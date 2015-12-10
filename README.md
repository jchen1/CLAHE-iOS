# Low-light-AR: Ghost-Detector App
Class project attempting to perform Augmented Reality under low-light conditions

Jeff Chen, Graeme Rock

#Summary

We plan to develop an iPhone application capable of running Augmented Reality (AR) under low-light conditions. We intend to use OpenCV to enhance low-light images taken with the camera with the eventual goal of being able to reliably detect interest points for 3D projections.

#Background (Introduction/Background/)
  A common motif in horror media is the idea that supernatural phenomena invisible to the naked eye (most commonly spirits) could be seen through photographic means. Over time, with the invention of digital cameras and their incorporation into smartphones, this idea has extended to such phenomena being viewable through digital means. We wish to emulate that experience using AR through the user’s smartphone.

There is a “Ghost Detector” apps currently in the iPhone and Android app stores currently consist of non-camera means of “detecting” ghosts, such as “radar,” EVP (Electronic Voice Phenomenon), and EMF Detectors (or real, in the case of the Mr. Ghost app, which uses an antenna).

#The Challenge
  Although interest point detection would be relatively trivial in normal lighting conditions, restricting our application to low-light adds significant hurdles. The iPhone camera is not particularly impressive under low-light, and there is no guarantee that the built-in flashlight would illuminate the scene well enough either. To make a standard interest point detector feasible, we will either have to process the camera images, find usable interest points in dark images, or both. Finally, naively using OpenCV to project 3D objects onto images probably will not perform well enough in real-time on mobile devices, so we may have to incorporate other libraries that utilize the GPU.

#Goals & Deliverables
  We plan to achieve relatively reliable interest point detection (and therefore 3D projection) under medium light conditions. If all goes well, we will extend this to low-light conditions. We plan to achieve this on still images, but if our processes are fast enough, we may be able to extend it to live video. Our success metric will be the application running video under medium or low light conditions, accurately projecting a 3D object within the room (nominally a ghost or other spooky creatures) within the room. 

#Schedule
  Since each step has a great deal of overlap in what we will both need to do, we will collaborate on each step according to what needs to be done by each Friday:

Friday   |  To Do / Done
-------- | ---------
Nov. 13  | Interest point detection working on iPhone / Found Kudan SDK
Nov. 20  | Lower light conditions / Testing Histogram Equalization
Nov. 27  | Augmented Reality working on iPhone / Found Kudan SDK
Dec. 4   | Test Graphics
Dec. 11  | Clean up loose ends (Due)

As we will use Git for version control so we do not need to check in with each other extensively, nor risk file loss. 

#References

Azuma, R.; Baillot, Y.; Behringer, R.; Feiner, S.; Julier, S.; MacIntyre, Blair, "Recent advances in augmented reality," in Computer Graphics and Applications, IEEE , vol.21, no.6, pp.34-47, Nov/Dec 2001
doi: 10.1109/38.963459
keywords: {augmented reality;annotation;augmented reality;complex equipment maintenance;complex equipment repair;error reduction strategies;medical visualization;path planning;registration;Augmented reality;Calibration;Cameras;Head;Laser beams;Lenses;Liquid crystal displays;Optical modulation;Retina;Virtual environment},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=963459&isnumber=20800

http://paranormal.about.com/od/ghosthuntinggeninfo/a/3-Paranormal-Apps-For-Android.htm

