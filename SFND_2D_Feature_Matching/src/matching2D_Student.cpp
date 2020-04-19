#include <numeric>
#include "matching2D.hpp"
#include "dataStructures.h"


using namespace std;
// time_keypoints ;
double t;
extern vector<time_keypoints> number_of_keypoints;
time_keypoints t_k;
             // Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      	
        //matcher = cv::FlannBasedMatcher::create();  
       if (descSource.type() != CV_32F)
        { 
         	descSource.convertTo(descSource, CV_32F);   //convert binary to float
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        cout << "# of matches = " << matches.size() << endl;
      
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
		int k = 2;
      	std::vector<std::vector<cv::DMatch>> knn_matches;
      	matcher->knnMatch(descSource, descRef, knn_matches, k);
        // ...
      double DistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < DistRatio * (*it)[1].distance)
            {
               
              matches.push_back((*it)[0]);
            }
//           	else{
//              cout <<  (*it)[0].distance << "and" << (*it)[1].distance << endl;
//             }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        cout << "# of matches = " << matches.size() << endl;
      	t_k.matched_keypoints = matches.size();
      	t_k.total_time = t_k.detectors_time + t_k.descriptors_time;

    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
  // -> BRIEF, ORB, FREAK, AKAZE, SIFT
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
		int bytes = 32;
      	bool use_orientation = false;
      
      	extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
        //...static Ptr<BriefDescriptorExtractor> cv::
    }
    
    else if (descriptorType.compare("ORB") == 0)
    {
		extractor = cv::ORB::create();
        //...
    }
  
  	else if (descriptorType.compare("FREAK") == 0)
    {
      
      	bool orientationNormalized = true;
		bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        const std::vector<int> & selectedPairs = std::vector<int>();
		
          extractor = cv::xfeatures2d::FREAK::create(); 
        //...
    }
  
    else if (descriptorType.compare("AKAZE") == 0)
    {
		
        int descriptor_type= cv::AKAZE::DESCRIPTOR_MLDB;
      	int descriptor_size=0;
        int descriptor_channels=3;
        float threshold=0.001f;
        int nOctaves=4; 
        int nOctaveLayers=4; 
        int diffusivity= cv::KAZE::DIFF_PM_G2;
     
      extractor = cv::AKAZE::create(); 

        //...
    }
  
    else if (descriptorType.compare("SIFT") == 0)
    {

      int nfeatures = 0;
      int nOctaveLayers = 3;
      double contrastThreshold = 0.04;
      double edgeThreshold = 10;
      double sigma = 1.6;
      
      extractor = cv::xfeatures2d::SIFT::create();	  
      //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    t_k.descriptors_time = 1000 * t / 1.0;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros( img.size(), CV_32FC1 );

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
	   t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);


    // visualize keypoints
     if (bVis)
    {
  	string windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
     }


}
void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;     // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  
  t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);
  
  cout << "added to number of keypoints" << endl;


   if (bVis)
    {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "FAST Results";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
   }
}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int thresh = 30;                                                             
  int octaves = 3;
  float patternScale = 1.0f; 
   
  cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(thresh, octaves, patternScale);

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);


   if (bVis)
    {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "BRISK Results";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
   }

  }

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
int 	nfeatures = 500;
float 	scaleFactor = 1.2f;
int 	nlevels = 8;
int 	edgeThreshold = 31;
int 	firstLevel = 0;
int 	WTA_K = 2;
int scoreType = (int)cv::ORB::HARRIS_SCORE;
int 	patchSize = 31;
int 	fastThreshold = 20 ;

//   cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold );
  
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

  
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);


   if (bVis)
    {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "ORB Results";
    cv::namedWindow(windowName, 8);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
   }
}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
int 	descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
int 	descriptor_size = 0;
int 	descriptor_channels = 3;
float 	threshold = 0.001f;
int 	nOctaves = 4;
int 	nOctaveLayers = 4;
int 	diffusivity = (int)cv::KAZE::DIFF_PM_G2; 
	
//  cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);

   cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);


   if (bVis)
    {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "AKAZE Results";
    cv::namedWindow(windowName, 3);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
   } 
}

void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
int 	nfeatures = 0;
int 	nOctaveLayers = 3;
double 	contrastThreshold = 0.04;
double 	edgeThreshold = 10;
double 	sigma = 1.6; 

 cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
  
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
t_k.detectors_number = keypoints.size();
    t_k.detectors_time = 1000 * t / 1.0;
    
    number_of_keypoints.push_back(t_k);

   if (bVis)
    {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "SIFT Results";
    cv::namedWindow(windowName, 9);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
   } 
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
// HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
  if (detectorType.compare("SHITOMASI") == 0)
    {
            detKeypointsShiTomasi(keypoints, img, false);
    	
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, img, false);
        }
  		else if (detectorType.compare("FAST") == 0)
        {
            detKeypointsFast(keypoints, img, false); 
        }

  		else if (detectorType.compare("BRISK") == 0)
        {
            detKeypointsBrisk(keypoints, img, false);           
        }
  
  		else if (detectorType.compare("ORB") == 0)
        {
            detKeypointsOrb(keypoints, img, false);          
        }
  
  		else if (detectorType.compare("AKAZE") == 0)
        {
            detKeypointsAkaze(keypoints, img, false);        
        }
  
  		else if (detectorType.compare("SIFT") == 0)
        {
            detKeypointsSift(keypoints, img, false);           
        }





}