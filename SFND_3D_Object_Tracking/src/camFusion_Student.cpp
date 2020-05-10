
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>


#include "camFusion.hpp"
#include "dataStructures.h"
#include "structIO.hpp"


using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
  std::vector<float> distance;
  float euclidean_mean;
   for(int i = 0; i < kptMatches.size(); i++)
  {

   
      if(boundingBox.roi.contains(kptsCurr[kptMatches[i].trainIdx].pt))
       {
          //boundingBox.kptMatches.push_back(kptsCurr[kptMatches[i]])
        //cout << col << endl;
            
             distance.push_back(sqrt(pow((kptsCurr[kptMatches[i].trainIdx].pt.x - kptsPrev[kptMatches[i].trainIdx].pt.x),2) + pow((kptsCurr[kptMatches[i].trainIdx].pt.y - kptsPrev[kptMatches[i].trainIdx].pt.y),2)));
       }
   }
  
  
  euclidean_mean = std::accumulate(distance.begin(),distance.end(),0.0) / distance.size();
   for(int i = 0; i < distance.size(); i++)
  {
    if(euclidean_mean/distance[i] > 0.7 && euclidean_mean/distance[i] < 1.3)
    {
    	boundingBox.kptMatches.push_back(kptMatches[i]);
    }
   }
       
     cout << "bounding box id" << boundingBox.boxID << "bB: " << boundingBox.kptMatches.size() << "kM" << kptMatches.size() << endl;
  
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
  vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
//       cout << "hi1" << endl;
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
//       cout << "hi2" << endl;

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK

cout << "TTC Camera: " << TTC << endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
  float d0 = 0; float d1 = 0; int i=0;int j=0; 
  std::vector<float> d0_vec;
  std::vector<float> d1_vec;
  //float average;
  float limit = 2000000000;
  for(auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); it1++)
  {
//     if(it1->x < limit)
//     {
      d0_vec.push_back(it1->x);
      limit = d0_vec.back();
      
    
    
    if(d0_vec.size() > 20)
    {
      	sort(d0_vec.begin(),d0_vec.end());
      	
//       	if (limit < d0_vec.back())
//         {
          //cout << "hiPrev" << endl;
          d0_vec.pop_back();
//         }
      d0 = accumulate(d0_vec.begin(),d0_vec.end(),0.0) / d0_vec.size();
//       cout << std::accumulate(d0_vec.begin(),d0_vec.end(),0.0) << " " << d0_vec.size() << endl;
    }
//     }     
//     cout << "prev" << lidarPointsPrev.size() << endl;

  }
  
    for(auto it1 = lidarPointsCurr.begin(); it1 != lidarPointsCurr.end(); it1++)
  {
//     if(it1->x < limit)
//     {
     	
      d1_vec.push_back(it1->x);
      limit = d1_vec.back();
//       cout << " x value " << it1->x << endl; 
    
    
    if(d1_vec.size() > 20)
    {
      	sort(d1_vec.begin(),d1_vec.end());
      	     	

//       	if (limit < d0_vec.back())
//         {
          //cout << "hiPrev" << endl;
          d1_vec.pop_back();
//         }
      d1 = accumulate(d1_vec.begin(),d1_vec.end(),0.0) / d1_vec.size();
//       cout << std::accumulate(d1_vec.begin(),d1_vec.end(),0.0) << " " << d1_vec.size() << endl;
    
    }
//       cout << "curr" << lidarPointsCurr.size() << endl;
  }
  
//   for(auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); it2++)
//   {
    
//     if(it2->x < limit)
//     {
     
//       d1_vec.push_back(it2->x);
//       limit = d1_vec[j];
      
//     }
    
//     if(it2 > lidarPointsCurr.begin() + 9)
//     {
//       	sort(d1_vec.begin(),d1_vec.end());
      	
//       	if (d1_vec[j] < d1_vec.back())
//         {
          
//           d1_vec.back() = d1_vec[j];
//         }
//       d1 = accumulate(d1_vec.begin(),d1_vec.end(),0.0) / d1_vec.size();
//     }
//     j++;
//   }
//   for(auto it1 = lidarPointsCurr.begin(); it1 != lidarPointsCurr.end(); it1++)
//   {
//     if(it1->x < limit)
//     {
//       d1 = it1->x;
//       limit = d1;
      
//     }
//   }
  
  	TTC = d1 / ((d0-d1)*frameRate);
//   	TTC = d1*frameRate / 0.0578;
  
  cout << "The TTC lidar is: " << TTC << " " << d0 << " " << d1 << " " << frameRate << endl; 
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  int col; int row; int col_counter; int maxCount;
  cv::Mat bbMatrix(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32SC1);
  bbMatrix = 0;
  // ...
  
//   loop pointer of matches/queryindex(
//     cv::Mat bbMatrix; size = boundingbox*boundingbox; 
    
//     derefp
//                       loop boundingbox (
//                         count matches.pt in all bounding boxes with roi in prev frame and in curr frame 
//                         prev frame 
//                         if currFrame and prevFrame matches are in boundingbox 
//   cout << "keypoint1 is" << (dataBuffer.end() - 1)->keypoints[matches[41].queryIdx].pt.x << " & "<<matches[41].queryIdx << " & " << (dataBuffer.end() - 1)->keypoints[matches[41].trainIdx].pt.x << " & "<<matches[41].trainIdx << " & "<<matches[41].imgIdx << endl;
// )
          //cout << "Matches " << matches.size() << endl;

  for(int i = 0; i < matches.size(); i++)
  {

    for(int j = 0; j < currFrame.boundingBoxes.size(); j++)
    {

      if(currFrame.boundingBoxes[j].roi.contains(currFrame.keypoints[matches[i].trainIdx].pt))
       {
          col = j;
        //cout << col << endl;
         break;
       }
       else
       { 
         continue;
       }
     }
       
      for(int j = 0; j < prevFrame.boundingBoxes.size(); j++)
    {
    if(prevFrame.boundingBoxes[j].roi.contains(prevFrame.keypoints[matches[i].queryIdx].pt))
       {
          row = j;

         break;
       }
       else
       { 
         continue;
       }
     }
 
//     cout << bbMatrix.at<int>(1,0) << " " << bbMatrix.at<int>(1,1) << " " << bbMatrix.at<int>(1,2) << " " << bbMatrix.at<int>(1,3) << " " << bbMatrix.at<int>(1,4) << " " << bbMatrix.at<int>(1,5) << " " << bbMatrix.at<int>(1,6) << " " << bbMatrix.at<int>(1,7) << " " << bbMatrix.at<int>(1,8) << " " << bbMatrix.at<int>(1,9) << " " << bbMatrix.at<int>(1,10) << " " << bbMatrix.at<int>(1,11) << " "  << endl;
    bbMatrix.at<int>(row,col) = bbMatrix.at<int>(row,col) + 1; 
//     bbMatrix(row,col) = bbMatrix.at<int>(row,col) + 1; 

  }

  	for(int row = 0; row < prevFrame.boundingBoxes.size(); row++)
    {
      for(int col = 0; col < currFrame.boundingBoxes.size(); col++)
      {
        if(bbMatrix.at<int>(row,col) > maxCount )
        {
          //bbBestMatches.insert(row,col);
          maxCount = bbMatrix.at<int>(row,col);
          col_counter = col;
          
        }
        
//         cout << "prevFrame.boundingBoxes.size() " << prevFrame.boundingBoxes.size() << endl;
//         cout << "currFrame.boundingBoxes.size() " << currFrame.boundingBoxes.size() << endl;        
//         //cout << bbMatrix.at<int>(row,col) << endl;
      }
      cout << row << " " << col_counter << " " << maxCount << endl;
      maxCount = 0;
      bbBestMatches[row] = col_counter;   
           
      
      
    }
      
      
//             for (auto iter = bbBestMatches.begin(); iter != bbBestMatches.end(); iter++)
// {
//     cout << "Key: " << iter->first << endl << "Values:" << iter->second << endl;
//             }
// //       for (int i = 0 ; i < currFrame.boundingBoxes.size(); i++){
// //         cout << "bbBestMatches[row] " << bbBestMatches.first << endl;
// //       }
//     
}
