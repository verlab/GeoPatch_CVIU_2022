  
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
void sampleKps(std::vector<cv::KeyPoint>& kps, std::vector<int> idxs)
{
	std::vector<cv::KeyPoint> new_kps;
	for(int i=0; i < kps.size(); i++)
		for(int j=0; j < idxs.size(); j++)
			if(kps[i].class_id == idxs[j])
				new_kps.push_back(kps[i]);

	kps = new_kps;
}

void run(Dict args)
{

	/********* Grabbing arguments **************/
	std::string dir =  args["-inputdir"][0];
	std::string file_ref = args["-refcloud"][0];
	std::string mode = args["-mode"][0];
	bool smooth = false;
	if(args["-smooth"][0] == "true")
		smooth = true;
	float scale = std::stof(args["-scale"][0]);
	float radius = std::stof(args["-radius"][0]);

	std::vector<cv::KeyPoint> keypoints_ref, keypoints_tgt;

	DepthProcessor dp_ref(dir,file_ref,smooth, scale, radius);
	CSVTable groundtruth_ref = CSVParser::loadKeypoints(dir + "/" + file_ref + ".csv", keypoints_ref);

	std::vector<int> valid_idxs;

	set_clock();
	dp_ref.getMesh()->extractGeodesicPaths(keypoints_ref);
	dp_ref.getMesh()->sampleGeodesicPoints();
	std::vector<cv::Mat> patches_ref = dp_ref.getMesh()->buildGeodesicPatches();

	printf("[Time] It took %.4f seconds to compute the geodesic patches for %d keypoints.\n", measure_clock(), keypoints_ref.size());
	printf("[Time] Total elapsed time: %.4f\n", total_time);

	if(mode == "patch")
		Utils::saveRawPatches(patches_ref, keypoints_ref, groundtruth_ref.size(), file_ref);

	// Example of how to save 3D plots
	//dp_ref.getMesh()->drawGeodesicSamples();
	//dp_ref.getMesh()->savePLY("plot_ref.ply"); 
	//dp_ref.getMesh()->drawGeodesicSamplesImage("plot_ref.png");

	//dp_ref.getMesh()->drawGeodesicGrid();
	//dp_ref.getMesh()->savePLY("plot_ref.ply"); 
	//exit(0);

	for(int cloud = 0; cloud < args["-clouds"].size(); cloud++)
	{
		std::string file_tgt = args["-clouds"][cloud];
		DepthProcessor dp_tgt(dir,file_tgt,smooth, scale, radius);

		CSVTable groundtruth_tgt = CSVParser::loadKeypoints(dir + "/" + file_tgt + ".csv", keypoints_tgt);

		//sampleKps(keypoints_tgt, valid_idxs);

		cv::Mat img_ref = cv::imread((dir + "/" + file_ref + "-rgb.png").c_str());
		cv::Mat img_tgt = cv::imread((dir + "/" + file_tgt + "-rgb.png").c_str());

		dp_tgt.getMesh()->extractGeodesicPaths(keypoints_tgt);
		dp_tgt.getMesh()->sampleGeodesicPoints();
		std::vector<cv::Mat> patches_tgt = dp_tgt.getMesh()->buildGeodesicPatches();
		
		//dp_tgt.getMesh()->drawGeodesicSamplesImage("plot_tgt.png");

		if(mode == "patch")
			Utils::saveRawPatches(patches_tgt, keypoints_tgt, groundtruth_tgt.size(), file_tgt);

		if(mode=="groundtruth")
		{

			cv::Mat patchviz_ref, patchviz_tgt;
			std::vector<cv::KeyPoint> keypointviz_ref, keypointviz_tgt;

			if(false) //Plot geodesic grid
			{
				if(file_tgt == "cloud_13")
				{
					//dp_tgt.getMesh()->drawGeodesicSamples();
					dp_tgt.getMesh()->drawGeodesicGrid();
					dp_tgt.getMesh()->savePLY("plot_tgt.ply"); 
					exit(0);
				} 
				else
					continue;
			}	


			std::vector<cv::DMatch> matches = Utils::calcAndSaveHammingDistancesNonrigid(keypoints_ref, keypoints_tgt, BRIEF::compute_rot(patches_ref),
			  BRIEF::compute_rot(patches_tgt), groundtruth_ref, groundtruth_tgt, file_ref + "__" + file_tgt + "__geopatchBRIEF.txt");
			
		}
	}

}

int main(int argc, char* argv[])
{

	ArgParser p(argc, argv); 

	run(p.get_args());

	printf("Done.\n");

	return 0;
}
