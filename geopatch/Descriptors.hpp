  
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%   This file is part of https://github.com/verlab/GeoPatch_CVIU_2022
//
//   geopatch-descriptor is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   geopatch-descriptor is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with geopatch-descriptor.  If not, see <http://www.gnu.org/licenses/>.
//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifndef BRIEF_H
#define BRIEF_H

#include <vector>
#include <random>
#include <math.h>

#include "opencv2/highgui.hpp"
//#include <opencv2/objdetect.hpp>

//@BRIEF: Couple of methods to compute descriptors of a given set of 'geodesic' patches (e.g., 32 x 32 patch)

namespace BRIEF
{

	std::vector<int> load_test_pairs(const std::string &filename) 
	{
		std::ifstream fin(filename.c_str());
		std::vector< std::vector<float> > test_pairs;
		std::vector<int> pattern;

		if (fin.is_open())
		{
			while(!fin.eof())
			{
				std::vector<float> r_and_theta(4);

				fin >> r_and_theta[0]; 
				fin >> r_and_theta[1]; // Point 1 (radius, theta);

				fin >> r_and_theta[2]; 
				fin >> r_and_theta[3]; // Point 2 (radius, theta)

				if(!fin.eof())
					test_pairs.push_back(r_and_theta);
			}

			fin.close();
			
			std::cout <<"Loaded " << test_pairs.size() << " pair tests from the file." << std::endl;
		}
		else
		{
			printf("\nERROR LOADING THE PAIRS TXT FILE...\n");
			exit(0);
		}


		for(int pair = 0 ; pair < test_pairs.size(); pair++)
		{	
			int r1 = std::min((int)test_pairs[pair][0], 31);
			int theta1 = (test_pairs[pair][1] * 180.0 / M_PI) / (360.0/32);
			int r2 = std::min((int)test_pairs[pair][2], 31);
			int theta2 = (test_pairs[pair][3] * 180.0 / M_PI) / (360.0/32);

			//printf("%d, %d, %d, %d\n", r1, theta1, r2, theta2) ; getchar();

			pattern.push_back(r1); pattern.push_back(theta1);
			pattern.push_back(r2); pattern.push_back(theta2);
		}


		return pattern;
	}

	std::vector<int> genPattern(int size, int patch_dims)
	{
		std::vector<int> pattern;

	    std::mt19937 mt1(45555555555), mt2(45555555555);
	    std::normal_distribution<float> Normal(0,(1/25.0) * pow(patch_dims/2,2)); 
	    std::uniform_real_distribution<float> Unif(0, patch_dims);
	    //std::uniform_real_distribution<float> Normal(0, 32);

	    for(int i=0; i < 2*size; i++)
	    {
	    	int u = (int)Unif(mt1);
	    	int n = (int)abs(Normal(mt2));
	    	if(n > patch_dims -1)
	    	{
	    		n = patch_dims - 1;
	    	}
			//printf("unif: %d, normal: %d\n", u, n);
			pattern.push_back(n);
			pattern.push_back(u);
	    }

		return pattern;

	}


	inline int smoothedSum(const cv::Mat& sum, float img_y, float img_x)
	{
		static const int HALF_KERNEL = 1;

		img_y = (int)(img_y + 0.5);
		img_x = (int)(img_x + 0.5);
		return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
			- sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
			- sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
			+ sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
	}	

	cv::Mat compute(std::vector<cv::Mat>& patches, std::vector<int> pattern = std::vector<int>())
	{
		//int descSize = 512;
		//std::vector<int> pattern = genPattern(descSize, patches[0].rows);
		if(pattern.size() == 0)
			pattern = load_test_pairs("/home/guipotje/Sources/2020-ijcv-geobit-extended-code/geobit/test_pairs_512.txt"); //genPattern(descSize, patches[0].rows);
		
		int descSize = pattern.size() / 4;
		cv::Mat descriptors;

		for(int i=0; i < patches.size(); i++)
		{
			cv::Mat keypoint_descriptor = cv::Mat::zeros(1, descSize/8, CV_8U);
			uchar* desc = keypoint_descriptor.ptr(0);
			uchar descByte = 0;
			int pos = 1;
			int t;

				// Construct integral image for fast smoothing (box filter)
			cv::Mat sum;
			cv::integral(patches[i], sum, CV_32S);


			for(t = 0; t < pattern.size(); t+=4)
			{
				if(pos % (8+1) ==0)
				{
					desc[(t/32) -1] = descByte;
					descByte = 0; pos = 1;
				}

				if(patches[i].at<uchar>(pattern[t], pattern[t+1]) < patches[i].at<uchar>(pattern[t+2], pattern[t+3]))
				//if( smoothedSum(sum, pattern[t], pattern[t+1]) < smoothedSum(sum, pattern[t+2], pattern[t+3]) )
					descByte = (uchar)1 << (8 - pos) | descByte;

				pos++;
			}
			
			desc[(t/32) -1] = descByte;

			descriptors.push_back(keypoint_descriptor);
		}

		return descriptors;
	}

	std::vector<cv::Mat> compute_rot(std::vector<cv::Mat>& patches, int nb_rots = 16)
	{
		std::vector<cv::Mat> rotated_descs;
		std::vector<int> pattern = 
		   genPattern(512, patches[0].rows);
		   //load_test_pairs("/home/guipotje/Sources/2020-ijcv-geobit-extended-code/geobit/test_pairs_512.txt");

		float incr = 32.0 / nb_rots;

		set_clock();

		for(float i=0; i < nb_rots; i++)
		{
			std::vector<int> rpattern = pattern;
			for(int j = 0; j < rpattern.size(); j+=4)
			{
				rpattern[j+1] = (rpattern[j+1] + (int)(i*incr)) % 32;
				rpattern[j+3] =(rpattern[j+3] + (int)(i*incr)) % 32;
			}

			rotated_descs.push_back(compute(patches, rpattern));
		}

		printf("Computing GeoBit took %.4f seconds.\n", measure_clock());

		return rotated_descs;
	}

};

namespace IntensityDesc
{
	cv::Mat compute(std::vector<cv::Mat>& patches)
	{
		cv::Mat descriptors;
		cv::Vec4d mean, stddev;

		for(int i=0; i < patches.size(); i++)
		{
			cv::Mat desc = patches[i].clone();
			desc.convertTo(desc, CV_32FC1);

			//normalization
			for(int y=0; y < 32; y+=8)
				for(int x = 0; x < 32; x+=8)
				{
					cv::Rect norm_window(x,y,8,8);
					cv::Mat roi = desc(norm_window);
					cv::meanStdDev(roi, mean, stddev);
					roi = (roi - mean[0])/stddev[0];
				}

			cv::resize(desc, desc, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
			desc = desc.reshape(1,1);
			
			//cv::meanStdDev(desc, mean, stddev);
			//desc = (desc - mean[0])/stddev[0];
			//cv::normalize(desc,desc,cv::NORM_L1);
			//cv::sqrt(desc, desc);
			descriptors.push_back(desc);
			//std::cout << desc << std::endl; exit(0);
		}


		return descriptors;
	}
};	

/*
namespace HOG
{
	cv::Mat compute(std::vector<cv::Mat>& patches)
	{	
		cv::HOGDescriptor hog(cv::Size(32, 32),cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1);
		cv::Mat descriptors;
		cv::Vec4d mean, stddev;

		for(int i=0; i < patches.size(); i++)
		{
			cv::Mat patch = patches[i];
			std::vector<float> desc;

			hog.compute( patch, desc, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
			cv::Mat fdesc = cv::Mat(desc).clone().t();
			//RootSIFT normalization for increased performance with L2 dist
			//cv::normalize(fdesc,fdesc,cv::NORM_L1);

			//cv::sqrt(fdesc, fdesc);		
			cv::meanStdDev(fdesc, mean, stddev);
			fdesc = (fdesc - mean[0])/stddev[0];

			descriptors.push_back(fdesc);
			//printf("%d\n", desc.size()); exit(0);
		}

		return descriptors;
	}
};	
*/

#endif