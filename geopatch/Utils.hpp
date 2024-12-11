  
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

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/features2d.hpp>

namespace Utils
{

	std::vector<cv::Mat> extract_cartesian(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img)
	{
		std::vector<cv::Mat> patches;

		for(int i=0; i < keypoints.size(); i++)
		{
			cv::Mat canvas(32, 32, CV_8UC1);
			int y = keypoints[i].pt.y;
			int x = keypoints[i].pt.x;
			//canvas = img(cv::Range(y - 16, y + 16), cv::Range(x - 16, x + 16));

			for(int y_p =- 16; y_p <  16; y_p++)
				for(int x_p= -16; x_p < 16; x_p++)
				{
					if(y_p+y >=0 && y_p+y < img.rows && x_p+x >=0 && x_p + x < img.cols )
						canvas.at<uint8_t>(16+y_p, 16+x_p) = img.at<uint8_t>(y_p+y, x_p+x);
				}

			patches.push_back(canvas);
		}

		//printf("Size cart patch = %d %d \n", patches[0].rows, patches[0].cols);
		return patches;
	}

	cv::Mat drawGroundTruth(cv::Mat& img_ref, std::vector<cv::KeyPoint>& keypoints_ref, cv::Mat& img_tgt, std::vector<cv::KeyPoint>& keypoints_tgt,
						std::vector<cv::DMatch>& matches, bool drawOnlyWrongs)
	{
		std::vector<cv::DMatch> corrects, incorrects;
		cv::Mat out;

		for(int i=0; i < matches.size(); i++)
		{
			cv::KeyPoint kp1 = keypoints_ref[matches[i].queryIdx];
			cv::KeyPoint kp2 = keypoints_tgt[matches[i].trainIdx];

			if(kp1.class_id == kp2.class_id)
				corrects.push_back(matches[i]);
			else
				incorrects.push_back(matches[i]);
		}

		cv::drawMatches(img_ref, keypoints_ref, img_tgt, keypoints_tgt, incorrects, out, cv::Scalar::all(-1));
		
		if(!drawOnlyWrongs)
			cv::drawMatches(img_ref, keypoints_ref, img_tgt, keypoints_tgt, corrects, out, cv::Scalar(0,255,0),
		                 cv::Scalar::all(-1), std::vector< char >(), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

		return out;
	}

	
	void saveRawPatches(std::vector<cv::Mat> patches, std::vector<cv::KeyPoint> keypoints, int max_kps, std::string filename)
	{
		std::vector<cv::Mat> output(max_kps);
		for(int i=0 ; i < keypoints.size(); i++)
			output[keypoints[i].class_id] = patches[i];

		int rows = patches[0].rows;
		int cols = patches[0].cols;
		
		cv::Size sz = {cols*max_kps, rows};
		cv::Mat canvas(sz,CV_8UC1,cv::Scalar::all(255));

		for(int i=0; i < max_kps; i++){
			if (output[i].rows > 0)
			{
				cv::Rect rect(cv::Point(i*cols, 0), output[i].size());
				output[i].copyTo(canvas(rect));
			}
		}
		cv::imwrite(filename + "_RAW_patches.png", canvas);
	}

	void saveSinglePatch(std::vector<cv::KeyPoint> keypoints, std::vector<cv::Mat> patches, int id, std::string filename, bool cartesian = false)
	{
		cv::Mat canvas;
		for(int i=0; i < keypoints.size(); i++)
		{
			if(keypoints[i].class_id == id)
			{
				canvas = patches[i]; break;
			}

		}

		cv::resize(canvas, canvas, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);

		char buf[200];

		if(cartesian)
		sprintf(buf, "gif_%s_cartpatch.png",filename.c_str());
		else
		sprintf(buf, "gif_%s_geopatch.png",filename.c_str());

		//printf("%s\n",buf); getchar();

		cv::imwrite(buf, canvas);
	}

	void saveGTPatches(std::vector<cv::KeyPoint> keypoints, std::vector<cv::Mat> patches, int max_kps, std::string filename,
		 std::vector<cv::KeyPoint>& new_keypoints, cv::Mat& patchviz, bool cartesian = false)
	{

		if(keypoints.size() == 0 || patches.size() == 0)
			return;

		std::vector<cv::Mat> output(max_kps);
		int cols = 8;
		cv::Mat canvas(((int)(max_kps/cols) + 1) * (patches[0].rows+2), (cols+1) * (patches[0].rows+2), CV_8UC1,cv::Scalar::all(255));

		for(int i=0 ; i < keypoints.size(); i++)
			output[keypoints[i].class_id] = patches[i];


		int kp = 0;
		for(int i=0; i < output.size(); i++)
		{
			int y = i / cols;
			int x = i % cols;

			cv::Point p0(x * (patches[0].rows+2), y * (patches[0].rows+2));
			cv::Rect dstRect(p0, output[i].size());
			if(output[i].rows > 0)
			{				
				output[i].copyTo(canvas(dstRect));
				char buf[250]; sprintf(buf, "%d", i);
				cv::putText(canvas, buf ,cv::Point2f(p0.x + 10, p0.y + 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 
				0.5, cv::Scalar(255), 0.6, CV_AA);
				cv::KeyPoint new_kp = keypoints[kp];
				new_kp.pt.x = p0.x +  patches[0].rows/2; new_kp.pt.y = p0.y +  patches[0].rows/2;
				new_kp.pt.x*=2; new_kp.pt.y*=2; 
				new_keypoints.push_back(new_kp);
				kp++;
			}

		}

		cv::resize(canvas, canvas, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);

		if(cartesian)
			cv::imwrite(filename + "_cartpatches.png", canvas);
		else
			cv::imwrite(filename + "_patches.png", canvas);


		patchviz = canvas;

	}



	std::vector<cv::DMatch> calcAndSaveDistances(std::vector<cv::KeyPoint> kp_query, std::vector<cv::KeyPoint> kp_tgt, 
	cv::Mat desc_query, cv::Mat desc_tgt, CSVTable query, CSVTable tgt, std::string file_name, int normType)
	 {
	   //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
	   std::vector<cv::DMatch> matches;
	   
	   std::ofstream oFile(file_name.c_str());
	   
	   oFile << query.size() << " " << tgt.size() << std::endl;
	   
	   cv::Mat dist_mat(query.size(),tgt.size(),CV_64F,cv::Scalar(-1));
	   
		int c_hits=0;

	   for(size_t i=0; i < desc_query.rows; i++)
	   {
		double menor = 99999;
		size_t menor_idx=-1, menor_i=-1, menor_j = -1;
		 
		for(size_t j = 0; j < desc_tgt.rows; j++)
		  {
			int _i = kp_query[i].class_id; //correct idx
			int _j = kp_tgt[j].class_id; //correct idx
			
			//if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
			//    std::cout << "Estouro: " << _i << " " << _j << std::endl;
			
			if(!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
			  continue;

			if(query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
			{
				auto start = std::chrono::steady_clock::now();
				dist_mat.at<double>(_i,_j) = cv::norm(desc_query.row(i), desc_tgt.row(j),normType);
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> diff = end-start;
				//matching_sum+= diff.count();

			  if(dist_mat.at<double>(_i,_j) < menor )
			  {
							menor = dist_mat.at<double>(_i,_j);
							menor_i = _i;
							menor_j = _j;
							menor_idx = j;
						}
			}
			
			//oFile << cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING) << " ";
		  }

			  cv::DMatch d;
			  d.distance = menor;
			  d.queryIdx = i;
			  d.trainIdx = menor_idx;

			if(d.queryIdx >=0 && d.trainIdx >=0)
			{
				matches.push_back(d);
				if(menor_i == menor_j)
					c_hits++;
			}

		}
		  
	  for(int i=0; i < dist_mat.rows;i++)
		for(int j=0; j < dist_mat.cols; j++)
		{
		  oFile << dist_mat.at<double>(i,j) << " ";
		}
		
		oFile << std::endl;   
		oFile.close(); 
		std::cout <<"Correct matches: " << c_hits << " of " << matches.size() << std::endl;
	   
	   
	   return matches;
	 }



int norm_hamming_nonrigid(std::vector<cv::Mat>& src, std::vector<cv::Mat>& tgt, int idx_d1, int idx_d2)
{
	std::vector<int> distances;

	for(int i=0; i < src.size(); i++)
		distances.push_back(cv::norm(src[0].row(idx_d1), tgt[i].row(idx_d2),cv::NORM_HAMMING));

	size_t min_idx =  std::distance(std::begin(distances),std::min_element(std::begin(distances), std::end(distances)));
		
	return distances[min_idx];
	
}

std::vector<cv::DMatch> calcAndSaveHammingDistancesNonrigid(std::vector<cv::KeyPoint> kp_query, std::vector<cv::KeyPoint> kp_tgt, 
std::vector<cv::Mat> desc_query, std::vector<cv::Mat> desc_tgt, CSVTable query, CSVTable tgt, std::string file_name)
 {
   //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
   std::vector<cv::DMatch> matches;
   
   std::ofstream oFile(file_name.c_str());
   
   oFile << query.size() << " " << tgt.size() << std::endl;
   
   cv::Mat dist_mat(query.size(),tgt.size(),CV_32S,cv::Scalar(-1));
   
	int c_hits=0;


   set_clock();

   for(size_t i=0; i < desc_query[0].rows; i++)
   {
	int menor = 999, menor_idx=-1, menor_i=-1, menor_j = -1;
	 
	for(size_t j = 0; j < desc_tgt[0].rows; j++)
	  {
		int _i = kp_query[i].class_id; //correct idx
		int _j = kp_tgt[j].class_id; //correct idx
		
		//if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
		//    std::cout << "Estouro: " << _i << " " << _j << std::endl;
		
		if(!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
		  continue;

		if(query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
		{
		  
		  dist_mat.at<int>(_i,_j) = norm_hamming_nonrigid(desc_query, desc_tgt, i, j);
		  if(dist_mat.at<int>(_i,_j) < menor )
		  {
						menor = dist_mat.at<int>(_i,_j);
						menor_i = _i;
						menor_j = _j;
						menor_idx = j;
					}
		}
		
		//oFile << cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING) << " ";
	  }

		  cv::DMatch d;
		  d.distance = menor;
		  d.queryIdx = i;
		  d.trainIdx = menor_idx;

		if(d.queryIdx >=0 && d.trainIdx >=0)
		{
			matches.push_back(d);
			if(menor_i == menor_j)
				c_hits++;
		}

	}

	printf("Took %.3f seconds to extract geobit\n", measure_clock());
	  
  for(int i=0; i < dist_mat.rows;i++)
	for(int j=0; j < dist_mat.cols; j++)
	{
	  oFile << dist_mat.at<int>(i,j) << " ";
	}
	
	oFile << std::endl;   
	oFile.close(); 
	std::cout <<"Correct matches: " << c_hits << " of " << matches.size() << std::endl;
   
   
   return matches;
 }
 


};

#endif