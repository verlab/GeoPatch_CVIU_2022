  
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

#include "Mesh.hpp"
#define WRITE_LOG 0

void Mesh::drawGeodesicPaths()
{
	for(int kp = 0; kp < keypoint_dirs.size(); kp++)
		for(int bin = 0 ; bin < keypoint_dirs[kp].size(); bin++)
		{
			for(int i=0; i < keypoint_dirs[kp][bin].size()-1; i++)
			{			
				Vec3 ab = keypoint_dirs[kp][bin][i+1] - keypoint_dirs[kp][bin][i];
				for(int j=0; j < 10 ; j++)
				{
					Vec3 pt = keypoint_dirs[kp][bin][i] + ab * (j/10.0);
					vertex vt(0,0,0);	
					vt.pt = pt;
					vt.r = 255;
					vertices.push_back(vt);
				}
			}

		}	
}

void Mesh::drawGeodesicSamples()
{

	for(int kp = 0; kp < geodesic_sampling.size(); kp++)
		for(int bin = 0 ; bin < geodesic_sampling[kp].size(); bin++)
		{
			for(int i=0; i < geodesic_sampling[kp][bin].size(); i++)
			{

				Vec3 pt = geodesic_sampling[kp][bin][i];

				for(float x = -0.5; x <= 0.5; x+= 0.25)
					for(float y = -0.5; y <= 0.5; y+= 0.25)
						for(float z = -0.5; z <= 0.5; z += 0.25)
						{
							vertex vt(x,y,z);	
							vt.pt = vt.pt + pt;
							vt.g = 255;
							vt.b = 255;
							vertices.push_back(vt);
						}
			}
		}		
		
}

void Mesh::drawGeodesicGrid()
{

	//first, draw the bins
	for(int kp = 0; kp < geodesic_sampling.size(); kp++)
		for(int bin = 0 ; bin < geodesic_sampling[kp].size(); bin++)
		{
			if((bin + 1)%4 == 0)
				for(int i=0; i < geodesic_sampling[kp][bin].size() - 1; i++)
				{
					Vec3 ab = geodesic_sampling[kp][bin][i+1] - geodesic_sampling[kp][bin][i];

					for(int j=0; j < 10 ; j++)
					{
						Vec3 pt = geodesic_sampling[kp][bin][i] + ab * (j/10.0);
						vertex vt(0,0,0);	
						vt.pt = pt;
						vt.pt.z -= 5;
						vt.r = 255;
						vertices.push_back(vt);
					}						
				}
		}		

	//then, draw the isocurves
	for(int kp = 0; kp < geodesic_sampling.size(); kp++)
		for(int i = 0 ; i < geodesic_sampling[kp][0].size(); i++)
		{
			if((i + 1)%8 == 0)
			{
				for(int bin =0; bin < geodesic_sampling[kp].size() - 1; bin++)
				{
					Vec3 ab = geodesic_sampling[kp][bin+1][i] - geodesic_sampling[kp][bin][i];

					for(int j=0; j < 20 ; j++)
					{
						Vec3 pt = geodesic_sampling[kp][bin][i] + ab * (j/20.0);
						vertex vt(0,0,0);	
						vt.pt = pt;
						vt.pt.z -= 5;
						vt.r = 255;
						vertices.push_back(vt);
					}						
				}

				//close the loop
				Vec3 ab = geodesic_sampling[kp][ 0 ][i] - geodesic_sampling[kp][ geodesic_sampling[kp].size()-1 ][i];
				for(int j=0; j < 20 ; j++)
				{
					Vec3 pt = geodesic_sampling[kp][geodesic_sampling[kp].size()-1][i] + ab * (j/20.0);
					vertex vt(0,0,0);	
					vt.pt = pt;
					vt.pt.z -= 5;
					vt.r = 255;
					vertices.push_back(vt);
				}	

			}				
		}
		
}

void Mesh::drawGeodesicSamplesImage(std::string out_img)
{
	cv::Mat canvas = image.clone();

	for(int kp = 0; kp < geodesic_sampling.size(); kp++)
	{
		cv::Mat patch(geodesic_sampling[kp].size(), geodesic_sampling[kp].size(), CV_8UC1,  cv::Scalar::all(0));

		for(int bin = 0; bin < geodesic_sampling[kp].size(); bin++)
		{
			for(int i=0; i < geodesic_sampling[kp][bin].size(); i++)
			{
				Vec3 p = geodesic_sampling[kp][bin][i];
				p = p / p.z;
				cv::Point2f p_img(p.x * cam.fx + cam.cx , p.y * cam.fy + cam.cy + 0.5);

				cv::circle(canvas, p_img, 1.0, cv::Scalar(255), -1);
			}
		}
	}

	cv::imwrite(out_img, canvas);
		
}


void Mesh::savePLY(std::string output_path)
{

   FILE *f = fopen(output_path.c_str(), "w");
   printf("Saving PLY in %s\n", output_path.c_str());

    size_t nb_faces = 0;
    for(size_t i=0; i < faces.size(); i++)
    	if(!(faces[i].v1 == 0 && faces[i].v2 == 0 && faces[i].v3 == 0))
    		nb_faces++;
	
	static char ply_header[] =
	"ply\n"
	"format ascii 1.0\n"
	"element vertex %ld\n"
	"property float x\n"
	"property float y\n"
	"property float z\n"
	"property uchar diffuse_red\n"
	"property uchar diffuse_green\n"
	"property uchar diffuse_blue\n"
	"element face %ld\n"
	"property list uchar int vertex_indices\n"
	"end_header\n";

	fprintf(f, ply_header, vertices.size(), nb_faces);

		for(size_t i=0 ; i < vertices.size(); i++)
		{

			fprintf(f,"%.2f %.2f %.2f %d %d %d\n",vertices[i].pt.x, vertices[i].pt.y, vertices[i].pt.z, 
														vertices[i].r, vertices[i].g, vertices[i].b);
		}

		for(size_t i=0; i < faces.size(); i++)
		{
			if(!(faces[i].v1 == 0 && faces[i].v2 == 0 && faces[i].v3 == 0))
				fprintf(f, "3 %ld %ld %ld\n", faces[i].v1, faces[i].v2, faces[i].v3);
		}

	fclose(f);	
	
}

std::vector<size_t> Mesh::getFaceIntersection(size_t v1, size_t v2)
{
	std::vector<size_t> faces(4);
	std::vector<size_t>::iterator it;

	//printf("set size %d\n", vertices[v1].faces_idx.size());
	//printf("set size2 %d\n", vertices[v2].faces_idx.size());

	it=std::set_intersection (vertices[v1].faces_idx.begin(), vertices[v1].faces_idx.end(),
	                          vertices[v2].faces_idx.begin(), vertices[v2].faces_idx.end(), faces.begin());
	faces.resize(it-faces.begin());

	return faces;
}


double angle(Vec3 v1, Vec3 v2)
{
	double angle = acos(v1.dot(v2)/(v1.norm()*v2.norm()));
	return angle * 180.0 / M_PI;
}


void Mesh::getP1P2(face& f, size_t v_idx, size_t& p1, size_t& p2)
{
	if(f.v1 == v_idx)
		{ p1 = f.v2; p2 = f.v3; }
	else if(f.v2 == v_idx)
		{ p1 = f.v1; p2 = f.v3; }
	else
		{ p1 = f.v1; p2 = f.v2; }	

}

std::vector< std::pair<Vec3, size_t> > Mesh::generateDirs(int n_dirs, size_t v_idx)
{
	std::vector< std::pair<Vec3,size_t> > dirs;

	float step = 360.0 / n_dirs;

	size_t face_idx = *vertices[v_idx].faces_idx.begin();
	face f = faces[face_idx]; //get first face

	size_t p1_idx,p2_idx;
	Vec3 p1, p2;

	getP1P2(f, v_idx, p1_idx, p2_idx);
	p1 = vertices[p1_idx].pt; p2 = vertices[p2_idx].pt;

	p1 = (p1 -vertices[v_idx].pt).unit();
	p2 = (p2 -vertices[v_idx].pt).unit();

	p1 = (p1 * 0.1234 + p2 * 0.8766).unit(); // start in the middle of a triangle to avoid vertex singularty

	int dirs_c = 0;
	double full_angle = angle(p1,p2), ratio, theta;
	int nb_bins = full_angle / step;
	double rest = step - fmod(full_angle, step);
	dirs.push_back(std::make_pair(p1, face_idx)); dirs_c++; //push the first orientation

	while(dirs_c < n_dirs)
	{ 	
		for(int i=1; dirs_c < n_dirs && i <= nb_bins; i++) // interpolate from p1 to p2
		{
			theta = (i * step) * M_PI / 180.0;
			Vec3 k = p1.cross(p2).unit();
			Vec3 d = p1 * cos(theta) + k.cross(p1) * sin(theta) + k * (k.dot(p1)) * (1.0 - cos(theta)); //Rodrigue's formula

			dirs.push_back(std::make_pair(d.unit(), face_idx));
			dirs_c++;
		}

		if(dirs_c == n_dirs) // break since we're finished
			break; 

		//get next face
		std::vector<size_t> twin_faces = getFaceIntersection(v_idx, p2_idx);
		if(twin_faces.size() == 2)
			face_idx == twin_faces[0] ? face_idx = twin_faces[1] : face_idx = twin_faces[0];
		else
			{printf("Keypoint on a hole!\n"); return std::vector< std::pair<Vec3,size_t> >();}

		f = faces[face_idx];
		p1_idx = p2_idx;

		//get next p2
		if(p1_idx == f.v2 && v_idx == f.v3 || p1_idx == f.v3 && v_idx == f.v2) p2_idx = f.v1;
		else if(p1_idx == f.v1 && v_idx == f.v3 || p1_idx == f.v3 && v_idx == f.v1) p2_idx = f.v2; 
		else p2_idx = f.v3;

		p1 = vertices[p1_idx].pt; p2 = vertices[p2_idx].pt;
		p1 = (p1 -vertices[v_idx].pt).unit();
		p2 = (p2 -vertices[v_idx].pt).unit();

		full_angle = angle(p1,p2);

		if(full_angle < rest) //skip that triangle, since its angle is smaller than the residual angle
			{nb_bins = 0; rest -= full_angle; continue;}

		theta = rest * M_PI / 180.0;
		Vec3 k = p1.cross(p2).unit();
		p1 = p1 * cos(theta) + k.cross(p1) * sin(theta) + k * (k.dot(p1)) * (1.0 - cos(theta));
		dirs.push_back(std::make_pair(p1, face_idx)); dirs_c++; //push the first orientation
		full_angle = angle(p1,p2);
		nb_bins = full_angle / step;
 		rest = step - fmod(full_angle, step);
	}

	return dirs;

}

void findC(const face& f, size_t a, size_t b, size_t& c)
{
		if(a == f.v2 && b == f.v3 || a == f.v3 && b == f.v2) c = f.v1;
		else if(a == f.v1 && b == f.v3 || a == f.v3 && b == f.v1) c = f.v2; 
		else c = f.v3;	
}

Vec3 rectPlaneIntersect(Vec3 p_n, Vec3 normal, Vec3 p_r, Vec3 dir)
{
	double r = (p_n - p_r).dot(normal) / dir.dot(normal);
	if(r < 0.0)
	{
		printf("WARNING: the intersection is behind dir! (negative r)\n");
	}

	return p_r + dir * r;
}

Vec3 Mesh::findPout(size_t& a, size_t& b, size_t c, Vec3 P, Vec3 V)
{
	
	if(V.unit() == (vertices[c].pt - P).unit())
	{
		printf("WARNING: The point goes out from a vertex!\n");
		return Vec3(0,0,0);
	}

	double sign = V.cross(vertices[a].pt - P).dot(V.cross(vertices[c].pt - P));
	if(sign < 0)
		std::swap(a,b);

	Vec3 bc = vertices[c].pt - vertices[b].pt;
	Vec3 f_normal = bc.cross(vertices[b].pt - vertices[a].pt);
	Vec3 bc_normal = f_normal.cross(bc);

	Vec3 Pout = rectPlaneIntersect(vertices[c].pt, bc_normal, P, V);

	//// Consistency Check /////
	Vec3 BPout = Pout - vertices[b].pt;

	if(!(bc.dot(BPout) > 0 && BPout.norm() <=  bc.norm() ))
		printf("WARNING: Found Pout is not on BC segment!\n");

	return Pout;
}

Vec3 Mesh::rotateDir(size_t a1, size_t b1, size_t c1, size_t a2, size_t b2, size_t c2, Vec3 dir)
{

	Vec3 n1 = (vertices[b1].pt - vertices[a1].pt).cross(vertices[c1].pt - vertices[a1].pt);
	Vec3 n2 = (vertices[b2].pt - vertices[a2].pt).cross(vertices[c2].pt - vertices[a2].pt);
	n1 = n1.unit(); n2 = n2.unit();

	if((vertices[c2].pt - vertices[a1].pt).dot(n1) < 0)
		n1 = - n1;
	if((vertices[a1].pt - vertices[c2].pt).dot(n2) < 0)
		n2 = - n2;

	Vec3 K = (n1.cross(n2)).unit();

	if(n1 == n2 || n1 == -n2)
		return dir;

	double sin =  (n1.unit().cross(n2.unit()) ).norm();
	double cos = n1.unit().dot(n2.unit());

	dir = dir.unit();

	Vec3 new_dir = dir * cos + K.cross(dir) * sin +  K * K.dot(dir) * (1.0 - cos);

	if(n2.dot(new_dir) > 1e-8)
	{
		printf("WARNING: the new rotated dir doesnt lie on the plane of the next face!\n");
		printf("%.9f\n", n2.dot(new_dir));
		//return Vec3(0,0,0);
	}
	else if( std::isnan(new_dir.x) || std::isnan(new_dir.y) || std::isnan(new_dir.y) ||
			 std::isinf(new_dir.x) || std::isinf(new_dir.y) || std::isinf(new_dir.z) )
	{
		printf("WARNING: the new rotated dir is not a number!\n");
	}
	
	return new_dir;
}

bool Mesh::findNextFace(size_t face1_idx, size_t p1, size_t p2, size_t& next_face)
{
	std::vector<size_t> twin_faces = getFaceIntersection(p1, p2);
	if(twin_faces.size() == 2)
	{
		if(twin_faces[0] == face1_idx)
			next_face = twin_faces[1];
		else
			next_face = twin_faces[0];
	}
	else 
		{ /*printf("Found a border!\n");*/ return false; }	

	return true;
}

std::vector<Vec3> Mesh::walkStraight(Vec3 dir, size_t v_idx, size_t f_idx, double max_path, int& status)
{
	std::vector<Vec3> path;
	size_t a,b,c, next_c;
	size_t face1_idx = f_idx, face2_idx;
	double amount_walked = 0;

	path.push_back(vertices[v_idx].pt);

	face f = faces[f_idx];
	getP1P2(f, v_idx, a, b);
	c = v_idx;

	Vec3 f_normal = (vertices[v_idx].pt - vertices[a].pt).cross(vertices[v_idx].pt - vertices[b].pt);
	Vec3 vb_normal = f_normal.cross(vertices[b].pt - vertices[a].pt);
	Vec3 P_now = rectPlaneIntersect(vertices[a].pt, vb_normal, vertices[v_idx].pt, dir); //first point going out an edge

	path.push_back(P_now); 
	amount_walked += (path[path.size() - 1] - path[path.size() -2]).norm();

	if(!findNextFace(face1_idx, a, b, face2_idx)) //get next face
		{printf("\rStopping the walk: %d", ++nborders); return path;}

	findC(faces[face2_idx], a, b, next_c); //find c given a,b of a triangle

	dir = rotateDir(c,a,b, a, b, next_c, dir);

	c = next_c;
	face1_idx = face2_idx;

	while(amount_walked < max_path)
	{
		P_now = findPout(a,b,c, P_now, dir);
		if(P_now.isNull())
		{		
			status = 1;
			return path;
		}

		path.push_back(P_now);
		amount_walked += (path[path.size() - 1] - path[path.size() -2]).norm();

		if(!findNextFace(face1_idx, b, c, face2_idx)) //get next face
			{printf("\rStopping the walk: %d", ++nborders); status = 2; return path;}

		findC(faces[face2_idx], b, c, next_c); //find c given a,b of a triangle

		dir = rotateDir(a, b, c, c, b, next_c , dir); 

		//if(dir.isNull())
		//	return path;

		face1_idx = face2_idx;
		a = c;
		c = next_c;

	}

	status = 0;
	return path;
}


void Mesh::extractGeodesicPaths(std::vector<cv::KeyPoint>& keypoints)
{
	keypoint_dirs = std::vector< std::vector< std::vector <Vec3> > >(keypoints.size());

	#pragma omp parallel for
	for(int i=0 ; i < keypoints.size() ; i++)
	{	
		int out_vertex = 0;
		size_t real_idx = (int)(keypoints[i].pt.x * scale) + (int)(keypoints[i].pt.y * scale) * cols; 
		size_t shifted = shifted_idx[real_idx];
		std::vector< std::pair<Vec3, size_t> > dirs = generateDirs(patch_size, shifted);
		//if(dirs.size() < patch_size)
		//{
		//	printf("Something went wrong.\n"); getchar();
		//}
		std::vector< std::vector < Vec3 > > paths;
		for(int bin =0 ; bin < dirs.size(); bin++)
		{
			int status;
			paths.push_back(walkStraight(dirs[bin].first,shifted, dirs[bin].second, patch_radius, status));
			if(status == 1)
				out_vertex++;
		}

		if(WRITE_LOG && out_vertex) // the walk path stopped in a vertex
		{
			#pragma omp critical
			{
				FILE *f = fopen("log.txt", "a");
				fprintf(f, "%d\n", out_vertex);
				fclose(f);
			}
		}

		keypoint_dirs[i] = paths;			
	}
}


void Mesh::sampleGeodesicPoints()
{
	double isocurve_size = patch_radius / (double) patch_size;

	geodesic_sampling = std::vector< std::vector< std::vector <Vec3> > >(keypoint_dirs.size());

	#pragma omp parallel for
	for(int kp = 0; kp < keypoint_dirs.size(); kp++)
	{

		std::vector< std::vector <Vec3> > sampled_points(patch_size);
		double walked, isocurve_id, curr_isocurve, len, ratio;

		for(int bin = 0 ; bin < keypoint_dirs[kp].size(); bin++)
		{		
			std::vector <Vec3> points(patch_size);
			isocurve_id = 1;
			walked = 0;

			for(int i=1; i < keypoint_dirs[kp][bin].size(); i++)
			{	
				Vec3 p1 =  keypoint_dirs[kp][bin][i-1];
				Vec3 p2 = keypoint_dirs[kp][bin][i];
				len = (p2 - p1).norm();

				curr_isocurve = isocurve_size * isocurve_id;

				while(isocurve_id -1 < patch_size && curr_isocurve <= walked + len)
				{
					ratio = (curr_isocurve - walked) / len;
					Vec3 mid_point = p1 + (p2 - p1) * ratio;

					points[(int)isocurve_id-1] = mid_point;
					isocurve_id++;

					curr_isocurve = isocurve_size * isocurve_id;
				}

				walked += len;
			}

			if(isocurve_id - 1 == 0) //treat the case where no points were sampled
			{
				points[(int)isocurve_id -1] = keypoint_dirs[kp][bin][0];
				isocurve_id++;
			}

			while(isocurve_id - 1 < patch_size)
			{
				points[(int)isocurve_id-1] = points[(int)isocurve_id-2];
				isocurve_id++; 
			}

			sampled_points[bin] = points;
		}

		geodesic_sampling[kp] = sampled_points;
	}	
}


inline int smoothedSum(const cv::Mat& img, const cv::Point& p, const int HALF_KERNEL)
{
	int x = p.x, y = p.y;
	cv::Point p1(std::max(0,x - HALF_KERNEL - 1), std::max(0,y - HALF_KERNEL -1));
	cv::Point p2(std::min(img.cols - 1, x + HALF_KERNEL), std::max(0,y - HALF_KERNEL -1) );
	cv::Point p3(std::max(0,x - HALF_KERNEL -1), std::min(img.rows - 1, y + HALF_KERNEL));
	cv::Point p4(std::min(img.cols - 1, x + HALF_KERNEL), std::min(img.rows - 1, y + HALF_KERNEL));

	cv::Point pn = p4 - p1;

	float n = (pn.x) * (pn.y);	

	int mean = (int) ((img.at<int>(p4) -  img.at<int>(p2) -  img.at<int>(p3) +  img.at<int>(p1)) / n + 0.5);

	return mean;
}


void Mesh::smoothAll(cv::Mat& img, int HALF_KERNEL)
{
	#pragma omp parallel for
	for(int y = 1; y < img.rows ; y++)
		for(int x = 1; x < img.cols; x++)
		{
			cv::Point p(x,y);
			image.at<unsigned char>(y-1,x-1) = smoothedSum(img, p, HALF_KERNEL);
		}

}

inline double bilinear(const float tx, const float ty,
				const double c00, const double c10, 
				const double c01, const double c11)
{ 
  //https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering

    double  a = c00 * (1 - tx) + c10 * tx; 
    double  b = c01 * (1 - tx) + c11 * tx; 
    return a * (1 - ty) + b * ty; 

} 

inline double bilinearInterp(const cv::Mat& scalarfield, float px, float py)
{
	double c00 = scalarfield.at<unsigned char>((int)py, (int)px);
	double c10 = scalarfield.at<unsigned char>((int)py, (int)px + 1);
	double c01 = scalarfield.at<unsigned char>((int)py + 1, (int)px);
	double c11 = scalarfield.at<unsigned char>((int)py + 1, (int)px + 1);

	return bilinear(px - (int)px, py - (int)py, c00, c10, c01, c11);
}


std::vector<cv::Mat> Mesh::buildGeodesicPatches()
{
	cv::Mat integral_img;
	std::vector<cv::Mat> patches(geodesic_sampling.size());

	cv::integral(image, integral_img);
	smoothAll(integral_img, 1);

	#pragma omp parallel for
	for(int kp = 0; kp < geodesic_sampling.size(); kp++)
	{
		cv::Mat patch(geodesic_sampling[kp].size(), geodesic_sampling[kp].size(), CV_8UC1,  cv::Scalar::all(0));

		for(int bin = 0; bin < geodesic_sampling[kp].size(); bin++)
		{
			for(int i=0; i < geodesic_sampling[kp][bin].size(); i++)
			{
				Vec3 p = geodesic_sampling[kp][bin][i];
				p = p / p.z;
				cv::Point2f p_img(p.x * cam.fx + cam.cx , p.y * cam.fy + cam.cy + 0.5);
				unsigned char v = (int)(bilinearInterp(image, p_img.x, p_img.y) + 0.5);

				patch.at<unsigned char>(i,bin) = v;

				//patch.at<unsigned char>(i,bin) = smoothedSum(integral_img, p_img, 1);
			}
		}

		patches[kp] = patch;
	}

	return patches;
}