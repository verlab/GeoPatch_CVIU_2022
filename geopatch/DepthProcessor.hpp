  
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

#ifndef DEPTHPROC
#define DEPTHPROC

#include "Mesh.hpp"

// OpenCV
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <queue> 
#include <numeric> 

//time measure
#include <chrono>
#include <ctime>

//Some one-liners that facilitates time measurements
extern std::chrono::system_clock::time_point wcts;
extern double total_time;
void set_clock();
double measure_clock();

class DepthProcessor
{
	Mesh* m;
	bool show_results, smooth;
	double scale_factor, patch_radius;

	public:
		DepthProcessor(const std::string &inputdir, const std::string &filename, bool smooth = true, double scale_factor = 0.5, double patch_radius = 75.0);

		~DepthProcessor(){delete m;}

		void interpolateDepth(cv::Mat& depth);
		void loadCloudFromPNGImages(const std::string &inputdir, const std::string &filename, cv::Mat& rgb, cv::Mat& matcloud);
		void buildMesh(cv::Mat& img, const cv::Mat& cloud);
		void applyRBFInterpolant(cv::Mat& depth, const std::vector<cv::Point>& known, const std::vector<cv::Point>& unknown);
		void applyIDWInterpolant(cv::Mat& depth, const std::vector<cv::Point>& known, const std::vector<cv::Point>& unknown);
		void nanResize(cv::Mat& img, float scale);

		Mesh* getMesh(){return m;}


};


#endif
