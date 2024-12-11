  
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

#include "DepthProcessor.hpp"
#include "CSVParser.hpp"
#include "Descriptors.hpp"
#include "Utils.hpp"
#include "ArgParser.hpp"

int myrandom (int i) { return std::rand()%i;}

bool loadKeypoints(const std::string &filename, std::vector<cv::KeyPoint> &keypoints)
{
    std::ifstream fin(filename.c_str());
    float x, y, size, angle;
    int n = 0;
    std::string line;

    if (fin.is_open())
    {
        std::getline(fin, line); //csv header
        while (std::getline(fin, line))
        {
            if (!line.empty())
            {
                std::stringstream ss;
                char *pch;

                pch = strtok((char *)line.c_str(), " ,");
                while (pch != NULL)
                {
                    ss << std::string(pch) << " ";
                    pch = strtok(NULL, " ,");
                }

                ss >> x >> y >> size >> angle;

                cv::KeyPoint kp(cv::Point2f(0, 0), 7.0); //6
                kp.pt.x = x;
                kp.pt.y = y;
                kp.angle = angle;
                kp.size = size;
                kp.class_id = n;
                kp.octave = 0.0;
                keypoints.push_back(kp);
            }
            n++;
        }

        fin.close();
    }
    else
    {
        printf("Unable to open ground truth csv file. Gonna use the detector algorithm.\n");
        return false;
    }

    return true;
}

void saveRotDesc(std::vector<cv::Mat> desc, std::string output){
    std::ofstream outfile;
    outfile.open(output);

    outfile <<  desc[0].rows << "," << desc.size() << "," << desc[0].cols << "\n";

    for(int r=0; r < desc[0].rows; r++){
        for(int i = 0; i < desc.size(); i++){
            for(int c=0; c < desc[i].cols; c++){
                outfile << unsigned(desc[i].at<uint8_t>(r,c));
                if( (i < desc.size() - 1) || (c < desc[i].cols - 1) ){
                    outfile << ",";
                }
            }
        }
        outfile << "\n";
    }

    outfile.close();
}

int main(int argc, char* argv[])
{
	/********* Grabbing arguments **************/
	std::string dir =  argv[1];
	std::string outdir =  argv[2];
	std::string kpdir =  argv[3];

	bool smooth = false;
	if(std::string(argv[4]) == "true")
		smooth = true;

	float scale = std::stof(argv[5]);
	float radius = std::stof(argv[6]);

	bool geobit = false;
	if(std::string(argv[7]) == "true")
		geobit = true;

	int n_clouds = std::atoi(argv[8]);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "###################### Settings ######################" << std::endl;
    std::cout << "Dir: " << dir << std::endl;
    std::cout << "Output: " << outdir << std::endl;
    std::cout << "Keypoints: " << kpdir << std::endl;
    std::cout << "Smooth: " << smooth << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    std::cout << "Radius: " << radius << std::endl;
    std::cout << "Use GEOBIT: " << geobit << std::endl;
    std::cout << "N Clouds: " << n_clouds << std::endl;
    std::cout << "######################################################" << std::endl;

	for(int cloud = 0; cloud < n_clouds; cloud++)
	{
    	std::vector<cv::KeyPoint> keypoints_tgt;
		std::string file_tgt = argv[9 + cloud ];

		DepthProcessor dp_tgt(dir,file_tgt,smooth, scale, radius);	
		bool load_ok = loadKeypoints(kpdir + "/" + file_tgt + ".kp", keypoints_tgt);
        if(!load_ok){
            std::cout  << "[I] Missing" << kpdir + "/" + file_tgt + ".kp" << std::endl;
            continue;
        }

		cv::Mat img_tgt = cv::imread((dir + "/" + file_tgt + "-rgb.png").c_str());

		dp_tgt.getMesh()->extractGeodesicPaths(keypoints_tgt);
		dp_tgt.getMesh()->sampleGeodesicPoints();
		std::vector<cv::Mat> patches_tgt = dp_tgt.getMesh()->buildGeodesicPatches();

        if(geobit){
            std::vector<cv::Mat> rot_desc = BRIEF::compute_rot(patches_tgt);
            saveRotDesc(rot_desc, outdir + "/" + file_tgt + ".desc");
        }else{
    		Utils::saveRawPatches(patches_tgt, keypoints_tgt, keypoints_tgt.size(), outdir + "/" + file_tgt);
        }

	}
    return 0;
}
