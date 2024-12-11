#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#define WSIZE 5
int c=0;
cv::Mat getCvMatAt(Image* img, int row, int col)
{
 row = img->sizeY - row - 1;
 cv::Mat cv_img = cv::Mat(WSIZE,WSIZE,CV_8UC3,cv::Scalar(127,127,127));

	for(int i= -WSIZE/2; i <= WSIZE/2 ; i++)
		for(int j= -WSIZE/2; j <= WSIZE/2; j++)
		{
			int _row = row + i;
			int _col = col + j;

			if(_row >= 0 && _row < img->sizeY && _col >= 0 && _col < img->sizeX)
			{	
				cv::Vec3b color;

				color[0] = img->data[_row*img->sizeX*4 + _col*4];
				color[1] = img->data[_row*img->sizeX*4 + _col*4 + 1];
				color[2] = img->data[_row*img->sizeX*4+  _col*4 + 2];

				cv_img.at<cv::Vec3b>(WSIZE -1 - (i + WSIZE/2),  (j+ WSIZE/2) ) = color;
			}
		}

	char buf[100];
	sprintf(buf,"estouro/%d.jpg",c++);
	//cv::imwrite(buf,cv_img);
	cv::Mat gray;
	cv::cvtColor(cv_img, gray, CV_RGB2GRAY);
	return gray;
}

double getHarrisScore(cv::Mat img)
{
	 /// Detector parameters
  //img.convertTo(img, CV_32F);
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  cv::Mat response_map;
  cv::Mat mean, var;
  cv::Mat laplacian;

  /// Harris
  cv::cornerHarris(img, response_map, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  response_map = response_map*1e6;
  response_map = cv::abs(response_map);
  double min, max = 1.0;
  cv::minMaxLoc(response_map, &min, &max);
  //std::cout << max << std::endl;
  return max;

  // Laplacian operator
  /*cv::Laplacian(img, laplacian, CV_32F, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::meanStdDev(laplacian, mean, var);
  double score =  var.at<double>(0);
  std::cout << score << std::endl;
  return score*score;
*/
}



std::vector<cv::KeyPoint> detectOrbKeypoints(cv::Mat& cv_image)
{	
	cv::Mat img;
	cv::flip(cv_image, img, 0);

	std::vector<std::vector<cv::KeyPoint> > kp_grid;
	int grid_height = 20, grid_width = 12; //20 , 12

	cv::KeyPoint k_template;
	k_template.response = 0;

	for(int i=0; i < grid_height; i++)
		kp_grid.push_back(std::vector<cv::KeyPoint>(grid_width, k_template));

	cv::Ptr<cv::FeatureDetector> detector =  cv::ORB::create(1300, 1, 1);
	std::vector<cv::KeyPoint> keypoints, new_keypoints;
	detector->detect(img, keypoints);

	for(int i=0; i < keypoints.size(); i++)
	{
		float x,y;
		x = (keypoints[i].pt.x / (img.cols-1)) * (grid_width-1);
		y = (keypoints[i].pt.y / (img.rows-1)) * (grid_height-1);

		if(kp_grid[(int)y][(int)x].response < keypoints[i].response)
			kp_grid[(int)y][(int)x] = keypoints[i];
	}

	for(int i=0; i < kp_grid.size(); i++)
		for(int j=0; j < kp_grid[i].size(); j++)
			if(kp_grid[i][j].response > 0)
				new_keypoints.push_back(kp_grid[i][j]);



	cv::KeyPointsFilter::retainBest(new_keypoints, 250);
	printf("Nb of keypoints: %ld\n", new_keypoints.size()); //getchar();

	cv::Mat buffer;
	cv::drawKeypoints(img, new_keypoints, img, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DEFAULT );
	//cv::imwrite("detectedkeypoints.png", buffer);
	//cv::flip(img, img, 0);

	return new_keypoints;

}