  
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

#ifndef CSVPARSER
#define CSVPARSER

//@BRIEF: Method to load a ground-truth keypoint file

#include <map>
#include <vector>

typedef std::vector<std::map<std::string,float> > CSVTable;

namespace CSVParser
{

	CSVTable loadKeypoints(const std::string &filename, std::vector<cv::KeyPoint> &keypoints)
	{
		CSVTable csv_data;
		keypoints.clear();
		std::ifstream fin(filename.c_str());
		int id, valid;
		float x,y;
		std::string line;

		if (fin.is_open())
		{
			std::getline(fin, line); //csv header
			while ( std::getline(fin, line) ) 
			{
				if (!line.empty()) 
				{   
					std::stringstream ss;
					char * pch;

					pch = strtok ((char*)line.c_str()," ,");
					while (pch != NULL)
					{
						ss << std::string(pch) << " ";
						pch = strtok (NULL, " ,");
					}

					ss >> id >> x >> y >> valid;
					
					if(x<0 || y<0)
						valid = 0;

		
					std::map<std::string,float> csv_line;
					csv_line["id"] = id;
					csv_line["x"] = x;
					csv_line["y"] = y;
					csv_line["valid"] = valid;
					csv_data.push_back(csv_line);

					//printf("loaded %d %d from file\n",x, y, valid);

					if(valid)
					{
						cv::KeyPoint kp(cv::Point2f(0,0), 12.0); //6 //7 
						kp.pt.x = x;
						kp.pt.y = y;
						kp.class_id = id;
						//kp.size = keypoint_scale;
						kp.octave = 0.0;
						keypoints.push_back(kp);
						csv_data[csv_data.size()-1]["idx"] = keypoints.size()-1;
					}
				}
			}

			fin.close();
		}
		else
		{ 
			printf("Unable to open ground truth csv file.\n"); 
		}

		printf("[Info] Loaded %d groundtruth keypoints.\n",keypoints.size());
		return csv_data;
	}

};

#endif