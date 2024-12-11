  
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

#ifndef MESH
#define MESH

#include <set>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <utility>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "Vec3.hpp"

//@Brief
//Face-oriented mesh data structure
//where each vertex points to the faces they're in, where one
//can efficiently get face intersection of an edge
struct face
{
	size_t v1, v2, v3;

	face(){ v1 = 0; v2 = 0; v3 = 0; }

	face(size_t _v1, size_t _v2, size_t _v3)
	{
		v1 = _v1;
		v2 = _v2;
		v3 = _v3;
	}


};

struct vertex
{
	Vec3 pt;
	unsigned char r, g, b; //Color information (intensity)
	std::set<size_t> faces_idx;

	vertex(){r = g = b = 0; }

	vertex(double _x, double _y, double _z)
	{
		pt.x = _x;
		pt.y = _y;
		pt.z = _z;

		r = g = b = 0;
	}
};

struct intrinsics
{
	double fx, fy, cx, cy;
};

class Mesh
{
	public:
		std::vector<vertex> vertices;
		std::vector<face> faces;
		std::vector<size_t> shifted_idx; //Maps keypoint (y,x) positions to vertex positions in the mesh
		std::vector< std::vector< std::vector <Vec3> > > keypoint_dirs; //geodesic paths
		std::vector< std::vector< std::vector <Vec3> > > geodesic_sampling; //sampled points in the paths

		cv::Mat image;
		intrinsics cam;
		size_t rows;
		size_t cols;

		int nborders;

		double scale; //scale factor if one wants to reduce the input images resolution (e.g., 0.5 reduces by half)
		int patch_size; // Desired patch size in pixels (e.g., 32 = 32 x 32 pixels)
		double patch_radius; //Desired patch radius in milimiters (e.g., 75 mm = 15 cm diameter)

		Mesh(){ scale = 0.5; patch_size = 32; patch_radius = 75; nborders=0; }
		~Mesh(){}

		/**** Drawing functions *****/
		void savePLY(std::string output_path);
		void drawGeodesicPaths();
		void drawGeodesicSamples();
		void drawGeodesicSamplesImage(std::string out_img);
		void drawGeodesicGrid();

		std::vector<size_t> getFaceIntersection(size_t v1, size_t v2);

		//@Brief
		//Calculates n_dirs equally spaced directions (in angles) for the sampling of the geodesic patch
		std::vector< std::pair<Vec3,size_t> > generateDirs(int n_dirs, size_t v_idx);
		void getP1P2(face& f, size_t v_idx, size_t& p1, size_t& p2);

		//@Brief
		//Definition of the functions used to calculates a straight walk on the manifold -- CPU version --
		//The main idea is that inside the triangles, the path is a line, and inbetween, the lines are rotated as if two adjacent triangles
		//were in the same plane -- which is a geodesic path --
		Vec3 rotateDir(size_t a1, size_t b1, size_t c1, size_t a2, size_t b2, size_t c2, Vec3 dir); // Uses Rodrigue's formula to rotate a dir between 2 neighbour triangles
		Vec3 findPout(size_t& a, size_t& b, size_t c, Vec3 P, Vec3 V); // Find the exit point of the path inside a face
		bool findNextFace(size_t face1_idx, size_t p1, size_t p2, size_t& next_face);
		std::vector<Vec3> walkStraight(Vec3 dir, size_t v_idx, size_t f_idx, double max_path, int& status);

		//Functions to compute the geodesic patch given the paths
		void smoothAll(cv::Mat& img, int HALF_KERNEL);
		void extractGeodesicPaths(std::vector<cv::KeyPoint>& keypoints);
		void sampleGeodesicPoints();
		std::vector<cv::Mat> buildGeodesicPatches();


};


#endif
