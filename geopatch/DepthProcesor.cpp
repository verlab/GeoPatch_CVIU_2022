  
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

std::chrono::system_clock::time_point wcts;
double total_time=0;

void set_clock() { wcts = std::chrono::system_clock::now(); }
double measure_clock(){ std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts); total_time+=wctduration.count(); return wctduration.count(); }

DepthProcessor::DepthProcessor(const std::string &inputdir, const std::string &filename, bool smooth, double scale_factor, double patch_radius)
{	
	show_results = false;
	this->smooth = smooth;
	this->scale_factor = scale_factor;
	this->patch_radius = patch_radius;

	printf("\n###################### Pre-processing mesh ######################\n");

	m = new Mesh();
	m->scale = this->scale_factor;
	m->patch_radius = this->patch_radius;

	cv::Mat cloud, img;
	loadCloudFromPNGImages(inputdir, filename, img, cloud);
	
	//set_clock();
	buildMesh(img, cloud);
	//printf("[Time] Building mesh took %.4f seconds.\n", measure_clock());
	printf("[Time] Total pre-processing time: %.4f seconds.\n", total_time);
	printf("#################################################################\n\n");

}

void nanMeanFilter(cv::Mat& img, int HALF_KERNEL)
{

	cv::Mat integral_img, integral_ones;
	cv::Mat mat_conv = img.clone();
	cv::Mat mat_ones = cv::Mat::ones(img.rows, img.cols, CV_64F);

	#pragma omp parallel for
	for(int i=0; i < mat_conv.rows; i++)
		for(int j=0; j < mat_conv.cols; j++)
		{
			 if( std::isnan(img.at<double>(i,j)) || img.at<double>(i, j) == 0)
			 {
				mat_conv.at<double>(i,j) = 0;
				mat_ones.at<double>(i,j) = 0;
			 }

		}	


	cv::integral(mat_conv, integral_img);
	cv::integral(mat_ones, integral_ones);

	for(int y=0; y < img.rows; y++)
		for(int x=0; x < img.cols; x++)
		{
			cv::Point p1(std::max(0,x - HALF_KERNEL -1), std::max(0,y - HALF_KERNEL -1));
			cv::Point p2(std::min(img.cols - 1, x + HALF_KERNEL), std::max(0,y - HALF_KERNEL -1) );
			cv::Point p3(std::max(0,x - HALF_KERNEL -1), std::min(img.rows - 1, y + HALF_KERNEL));
			cv::Point p4(std::min(img.cols - 1, x + HALF_KERNEL), std::min(img.rows - 1, y + HALF_KERNEL));

			double s1 = integral_img.at<double>(p4) -  integral_img.at<double>(p2) -  integral_img.at<double>(p3) +  integral_img.at<double>(p1);
			double s2 = integral_ones.at<double>(p4) -  integral_ones.at<double>(p2) -  integral_ones.at<double>(p3) +  integral_ones.at<double>(p1);

			if(s2 > 0 && img.at<double>(y,x) > 0)
				img.at<double>(y,x) = s1/s2;
		}

}

void nanConv(cv::Mat& img)
{    
	cv::Mat kernelX = cv::getGaussianKernel(7, 6.0, CV_64F );
	cv::Mat kernelY = cv::getGaussianKernel(7, 6.0, CV_64F );
	cv::Mat kernel = kernelX * kernelY.t();

	cv::Mat mat_ones = cv::Mat::ones(img.rows, img.cols, CV_64F);
	cv::Mat mat_conv = img.clone();

	//#pragma omp parallel for
	for(int i=0; i < mat_conv.rows; i++)
		for(int j=0; j < mat_conv.cols; j++)
		{
			 if( std::isnan(img.at<double>(i,j)) || img.at<double>(i, j) == 0)
			 {
				mat_conv.at<double>(i,j) = 0;
				mat_ones.at<double>(i,j) = 0;
			 }

		}

	cv::Mat conv1;
	cv::Mat conv2;
	cv::Mat smoothed;

	cv::filter2D(mat_conv, conv1, -1, kernel, cv::Point(-1,-1), 0,cv::BORDER_DEFAULT);
	cv::filter2D(mat_ones, conv2, -1, kernel, cv::Point(-1,-1), 0,cv::BORDER_DEFAULT);

	//conv1.setTo(0, conv1 <= 1e-20);
	//conv2.setTo(0, conv2 <= 1e-20);

	smoothed = conv1 / conv2;

	img = smoothed;
}

void DepthProcessor::nanResize(cv::Mat& img, float scale)
{
	//nanConv(img);
	float orig_size_rows = img.rows, orig_size_cols = img.cols;

	if(smooth)
	{
		for(int i=0; i < 2 ; i++)
		{
			nanMeanFilter(img, 3); 

			#pragma omp parallel for
			for(int i=0; i < img.rows; i++)
				for(int j=0; j < img.cols; j++)
					 if( img.at<double>(i, j) == 0)
						img.at<double>(i,j) = std::numeric_limits<double>::quiet_NaN();

			cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
		}	

	}

	cv::resize(img, img, cv::Size(orig_size_cols*scale, orig_size_rows*scale), cv::INTER_CUBIC);
}

void DepthProcessor::applyIDWInterpolant(cv::Mat& depth, const std::vector<cv::Point>& known, const std::vector<cv::Point>& unknown)
{
	#pragma omp parallel for
	for(size_t i=0; i < unknown.size(); i++)
	{	
		float idw[known.size()];
		double dist, dist_sum, value;
		float power = 2.3;
		dist_sum = value = 0;
		for(size_t j=0; j < known.size(); j++)
		{
			dist = cv::norm(unknown[i] - known[j]);
			if(dist > 0)
			{
				idw[j] = 1.0/pow(dist, power);
				dist_sum += idw[j];
			}

		}

		for(size_t j=0; j < known.size(); j++)
			value += idw[j] * depth.at<double>(known[j]);
		
		if(dist_sum > 0)
			depth.at<double>(unknown[i]) = value / dist_sum;
	}

}

double multiquadratic(double r, double e)
{
	return sqrt(1.0f + pow(e*r, 2));
}

void DepthProcessor::applyRBFInterpolant(cv::Mat& depth, const std::vector<cv::Point>& known, const std::vector<cv::Point>& unknown)
{
	cv::Mat A(known.size(), known.size(), CV_32F);
	cv::Mat b(known.size(), 1, CV_32F);
	cv::Mat x;
	
	float r;
	
	for(size_t i=0; i < A.rows; i++)
		for(size_t j=0; j < A.cols; j++)
		{
			r = cv::norm(known[i] - known[j]);
			A.at<float>(i,j) = multiquadratic(r, 1.0);
		}
			
	for(size_t i=0; i < b.rows; i++)
	{
		b.at<float>(i,0) = depth.at<double>(known[i]);
	}
		
	/*solve linear system Ax = b */
	
	if(!cv::solve(A,b,x, cv::DECOMP_NORMAL))
	{
		printf("Oops, exploded in solving linear system\n"); getchar();
	}
	
	cv::Mat dists(1, known.size(), CV_32F), result;
	
	/*interpolate depth for each pixel */
	for(size_t i=0; i < unknown.size(); i++)
		for(size_t j=0; j < known.size(); j++)
		{
			r =  cv::norm(unknown[i] - known[j]);
			dists.at<float>(0,j) = multiquadratic(r, 1.0);
			result = dists * x;
			//printf("result size: %d %d\n", result.rows, result.cols); getchar();
			depth.at<double>(unknown[i]) = result.at<float>(0,0);
		}
	
}


void DepthProcessor::interpolateDepth(cv::Mat& depth)
{
	cv::Mat out = depth.clone();
	cv::threshold( out, out, 0, 255, 0);
	out.convertTo(out, CV_8UC1);
	int perimeter_thresh = 400;
	cv::Mat canvas = out.clone();

	cv::Mat canvas2(out.rows, out.cols, CV_8UC1, cv::Scalar::all(255));
	
	canvas2 = depth.clone();
	canvas2 *= 50/1000.0;
	canvas2.convertTo(canvas2, CV_8UC1);

	if(show_results)
		cv::imwrite("/homeLocal/guipotje/depth1.png", canvas2);	

	std::vector< std::vector<cv::Point> > contours, small_contours, knowns, unknowns;
	cv::findContours(out, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	for(size_t i=0; i< contours.size(); i++) // filter big holes
		if(contours[i].size() < perimeter_thresh)
			small_contours.push_back(contours[i]);

	if(show_results)
		cv::imwrite("/homeLocal/guipotje/depth.png", out);
	cv::drawContours(canvas, small_contours, -1, cv::Scalar(127), 1, cv::LINE_8);
	if(show_results)
		cv::imwrite("/homeLocal/guipotje/depth_c.png", canvas);

	bool found;
	cv::Point seed;
	int qtd = 0;
	cv::Point dirs[] = {cv::Point(-1,0), cv::Point(1,0), cv::Point(0,-1), cv::Point(0,1)};

	for(size_t i=0; i < small_contours.size(); i++)
	{	
		found = false;
		for(size_t cidx = 0; cidx < small_contours[i].size() && !found; cidx++)
		{
			cv::Point p = small_contours[i][cidx];
			for(int j=-1; j <=1; j++)
				for(int k=-1; k <=1; k++)
				{	
					cv::Point dir(j, k), nb;
					nb = p + dir;

					if( !found && nb.x >=0 && nb.x < out.cols && nb.y >=0 && nb.y < out.rows &&
						cv::pointPolygonTest(small_contours[i], nb, false) > 0)
					{
						found = true;
						seed = nb;
					}
					
				}
		}

		if(found && out.at<unsigned char>(seed) == 0) // actual holes
		{	
			std::queue<cv::Point> q;
			cv::Point p;
			std::vector<cv::Point> contour = small_contours[i], fill;

			for(int ii=0; ii < contour.size(); ii++)
				canvas.at<unsigned char>(contour[ii]) = 200;

			q.push(seed);

			while(!q.empty())
			{
				p = q.front(); q.pop();
				if(canvas.at<unsigned char>(p) == 0)// unknown
				{
					fill.push_back(p); 
					canvas.at<unsigned char>(p) = 100;
					for(int ii = 0; ii < 4; ii++)
						q.push(p + dirs[ii]);
				}
				else if(canvas.at<unsigned char>(p) == 255 || canvas.at<unsigned char>(p) == 127) //known
				{
					contour.push_back(p);
					canvas.at<unsigned char>(p) = 100;
					for(int ii = 0; ii < 4; ii++)
						q.push(p + dirs[ii]);
				}
				
			}
				knowns.push_back(contour);
				unknowns.push_back(fill);
			//holes.push_back(small_contours[i]);
		}
	}

	canvas2 = cv::Mat(out.rows, out.cols, CV_8UC1, cv::Scalar::all(255));
	for(size_t i=0; i < knowns.size(); i++)
		for(size_t j=0; j < knowns[i].size(); j++)
			canvas2.at<unsigned char>(knowns[i][j]) = 0;
			
	for(size_t i=0; i < unknowns.size(); i++)
		for(size_t j=0; j < unknowns[i].size(); j++)
			canvas2.at<unsigned char>(unknowns[i][j]) = 127;	
			
			
	printf("[Info] Number of filled holes: %d\n", knowns.size());

	for(size_t i=0; i < knowns.size(); i++)
	{	
		applyIDWInterpolant(depth, knowns[i], unknowns[i]);
		//printf("Interpolated %d\n", i);
	}
		
	//canvas = out.clone();
	//cv::drawContours(canvas, holes, -1, cv::Scalar(127), 1, cv::LINE_8);
	if(show_results)
	{
		cv::imwrite("/homeLocal/guipotje/depth_h.png", canvas);
		cv::imwrite("/homeLocal/guipotje/hole_paintings.png", canvas2);
	}
	

	canvas2 = depth.clone();
	canvas2 *= 50/1000.0;
	canvas2.convertTo(canvas2, CV_8UC1);

	if(show_results)
		cv::imwrite("/homeLocal/guipotje/depth2.png", canvas2);

}

void DepthProcessor::loadCloudFromPNGImages(const std::string &inputdir, const std::string &filename, cv::Mat& rgb, cv::Mat& matcloud)
{
	set_clock();
	printf("Loading cloud from %s/%s\n", inputdir.c_str(), filename.c_str());
	cv::FileStorage fs;
	cv::Mat in_depth;
	cv::Mat K;

	fs.open(inputdir + "/intrinsics.xml", cv::FileStorage::READ);
	fs["intrinsics"] >> K; 

	//std::cout << K << std::endl;

	rgb = cv::imread(inputdir + "/" + filename + "-rgb.png", 0);
	in_depth = cv::imread(inputdir + "/" + filename + "-depth.png", cv::IMREAD_ANYDEPTH);
	in_depth.convertTo(in_depth, CV_64F);
	//in_depth = in_depth / 1000.0;

	printf("[Time] Loading RGB and Depth images from disk (%d x %d) took %.4f seconds.\n", in_depth.cols, in_depth.rows, measure_clock());

	set_clock();
	interpolateDepth(in_depth);
	printf("[Time] Filling holes took %.4f seconds. \n", measure_clock());


	set_clock();
	nanResize(in_depth, m->scale);
	printf("[Time] Smoothing and resizing took %.4f seconds.\n", measure_clock());
	

	if(show_results)
	{
		cv::Mat out_depth = in_depth.clone();
		out_depth*=50/1000.0;
		cv::imwrite("/homeLocal/guipotje/final_depth.png", out_depth);
	}

	set_clock();
	K.convertTo(K, CV_64F);
	K *= m->scale;
	K.at<double>(2,2) = 1.0;
	//cv::Mat Kinv = K.inv();

	matcloud = cv::Mat(in_depth.rows, in_depth.cols, CV_64FC3);
	double fx,fy,cx,cy;
	fx = K.at<double>(0,0);
	fy = K.at<double>(1,1);
	cx = K.at<double>(0,2);
	cy = K.at<double>(1,2);

	m->cam.fx = fx;  m->cam.fy = fy;
	m->cam.cx = cx;  m->cam.cy = cy;

	//printf(">> Projecting depth...\n");
	#pragma omp parallel for
	for (int y = 0 ; y < matcloud.rows; y++)
		for(int x = 0 ; x < matcloud.cols ; x++)
		{
			double px, py;
			double x3d, y3d, z3d;

			px = (x - cx) / fx;
			py = (y - cy) / fy;
			
			z3d = in_depth.at<double>(y,x);

			if(z3d != 0 && !std::isnan(z3d))
			{
				x3d= px * z3d;
				y3d= py * z3d;
			}
			else
				x3d = y3d = z3d = std::numeric_limits<double>::quiet_NaN();

		   matcloud.at<cv::Vec3d>(y,x)[0] = x3d;
		   matcloud.at<cv::Vec3d>(y,x)[1] = y3d;
		   matcloud.at<cv::Vec3d>(y,x)[2] = z3d;
		}

	printf("[Time] Re-projecting 3D point cloud (%d x %d), took %.4f seconds.\n", matcloud.cols, matcloud.rows, measure_clock());

}


void DepthProcessor::buildMesh(cv::Mat& img, const cv::Mat& cloud)
{
    size_t c = 0, i=0;

    set_clock();

    m->shifted_idx = std::vector<size_t>(cloud.rows * cloud.cols);
    m->vertices = std::vector<vertex>(cloud.rows * cloud.cols);
    m->faces = std::vector<face>(cloud.rows * cloud.cols * 2);
    cv::Mat nanMask(cloud.rows, cloud.cols, CV_8UC1, cv::Scalar::all(1));

    m->rows = cloud.rows;
    m->cols = cloud.cols;

    cv::resize(img, img, cloud.size(), cv::INTER_LINEAR);
    m->image = img;

	for(size_t y=0; y < cloud.rows; y++)
		for(size_t x = 0; x < cloud.cols; x++)
		{
			m->shifted_idx[i] = c;
			i++;
			bool isNaN = std::isnan(cloud.at<cv::Vec3d>(y,x)[0]) || 
			std::isnan(cloud.at<cv::Vec3d>(y,x)[1]) || 
			std::isnan(cloud.at<cv::Vec3d>(y,x)[2]);

			if(isNaN)
				nanMask.at<unsigned char>(y,x) = 0;

			if(!isNaN)
			{
				m->vertices[c] = (vertex(cloud.at<cv::Vec3d>(y,x)[0],
										  cloud.at<cv::Vec3d>(y,x)[1],
										  cloud.at<cv::Vec3d>(y,x)[2]));

				m->vertices[c].r = m->vertices[c].g = m->vertices[c].b = img.at<unsigned char>(y,x);
				c++;
			}

		}


	printf("[Time] Initializing vertices took %.4f seconds.\n", measure_clock());

	set_clock();
	//size_t f = 0;

	for(int shiftx = 0; shiftx < 2 ; shiftx++) //Trick to allow full parallelization without concurrent writing
		for(int shifty = 0; shifty < 2 ; shifty++)
		{
			#pragma omp parallel for
			for(size_t y = shifty; y < cloud.rows -1; y+=2)
				for(size_t x = shiftx; x < cloud.cols -1; x+=2)
				{
					size_t f = y * cloud.cols * 2 + x * 2;
					size_t f1, f2, f3;
					f1 = m->shifted_idx[x + cloud.cols * y];
					f2 = m->shifted_idx[(x+1) + cloud.cols * y];
					f3 = m->shifted_idx[x + cloud.cols * (y+1)];

					bool isNumber = nanMask.at<unsigned char>(y,x) &&
								 	nanMask.at<unsigned char>(y,x+1) &&
								 	nanMask.at<unsigned char>(y+1,x);

					if(isNumber)
					{
						m->faces[f] = (face(f1,f2,f3));
						m->vertices[f1].faces_idx.insert(f);
						m->vertices[f2].faces_idx.insert(f);
						m->vertices[f3].faces_idx.insert(f); f++;
					}

					f1 = m->shifted_idx[(x+1) + cloud.cols * (y+1)];
					f2 = m->shifted_idx[(x+1) + cloud.cols * y];
					f3 = m->shifted_idx[x + cloud.cols * (y+1)];

					isNumber = nanMask.at<unsigned char>(y+1,x+1) &&
							   nanMask.at<unsigned char>(y,x+1) &&
							   nanMask.at<unsigned char>(y+1,x);

					if(isNumber)
					{
						m->faces[f] = (face(f1,f2,f3));
						m->vertices[f1].faces_idx.insert(f);
						m->vertices[f2].faces_idx.insert(f);
						m->vertices[f3].faces_idx.insert(f); f++;
					}

				}

			}

		m->vertices.resize(c);
		//m->faces.resize(f);

		//printf("%ld - %ld\n", m->vertices.size(), m->faces.size());
		printf("[Time] Initializing and indexing faces took %.4f seconds.\n", measure_clock());

}
