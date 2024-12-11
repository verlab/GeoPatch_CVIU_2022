/*
History:
	2 Jun 2009 - Initial version
	6 Jan 2010 - Typo corrected in call of glutInitDisplayMode to enable depth-buffer (Thanks Martijn The)
	---
	03 Jul 2018 - Added texture support and tweaked with the parameters to allow isometric deformation by Potje
	04 Jul 2018 - Generate the RGB-D data with the OpenGL depth buffer by Potje

A good portion of this source code is accompanying the Cloth Tutorial at the cg.alexandra.dk blog. Many thanks to Jesper Mosegaard.

You may use the code in any way you see fit. Please leave a comment on the blog 
or send me an email if the code or tutorial was somehow helpful.

Everything needed is defined in this file, it is probably best read from the 
bottom and up, since dependancy is from the bottom and up


A short overview of this file is;
* includes
* physics constant

* class Vec3
* class Particle (with position Vec3)
* class Constraint (of two particles)
* class Cloth (with particles and constraints)

* Cloth object and ball (instance of Cloth Class)

* OpenGL/Glut methods, including display() and main() (calling methods on Cloth object)

* Texturing of the cloth from a Bitmap 
* Depth map calculation from OpenGL's depth buffer
* Joint display of the RGB and depth image for checking the simulation

Jesper Mosegaard, clothTutorial@jespermosegaard.dk

Tested on: Windows Vista / Visual Studio 2005
		   Linux (Red Hat) / GCC 4.1.2

*/

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <random>
#include <cstdint>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <string>

#include <Eigen/Dense> /*To project a 3D point to camera coordinates*/
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core.hpp>

typedef Eigen::Matrix<float,6,1> Vector6f;

/* Some physics constants */
#define DAMPING 0.01                   //0.01 // how much to damp the cloth simulation each frame
#define TIME_STEPSIZE2 0.5 * 0.5 * 0.7 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 200//25       //40 //40//15 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft and elastic)

int save_count = 0;
bool save_result = false;
bool _save_result = false;
//angle of rotation
float xpos = -13.3/2, ypos = 5, zpos = -7, xrot = 0, yrot = 0, zrot = 0, angle = 0.0;
float lastx, lasty;

long int global_time;
int variation_interval, wait_time;
int save_interval, save_range;
int max_frames;
float rot_increment;
float max_rot;
std::string verbose;

float light_variation;
float fx_var, fy_var, fz_var;
float fx, fy, fz;
float _fx, _fy, _fz;
float rot_var;
float cx, cy, xpos_off = 0, ypos_off = 0;
float w_half = 13.3/2, h_half = -10.0/2;
float gravity = 1.;

long Width;
long Height;

std::string out_dir, background;

cv::Mat gridmat;
std::vector<cv::KeyPoint> keypoints;

int save_sim;

std::mt19937 mt;
std::uniform_real_distribution<double> unif(0, 1);

//positions of the cubes

/* Texture handler*/

GLuint texture[3];

struct Image
{

    int sizeX;
    int sizeY;
    GLubyte *data;
};

Image *image1, *img_bg;

typedef struct Image Image;


//#include "output_generator.hpp"
#include "corner_score.hpp"

float randRange(float min, float max)
{
    return min + unif(mt) * (max - min);
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}



int ImageLoad(const char *filename, Image *image, bool detect)
{
    cv::Mat cv_image = cv::imread(std::string(filename));
    // cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    cv::flip(cv_image, cv_image, 0);

    if(detect)
        keypoints = detectOrbKeypoints(cv_image);

    if(verbose == "true")       
        printf("Detected %ld keypoints.\n", keypoints.size());

    if(!cv_image.data)                              // Check for invalid input
    {
        printf("Could not open or find the image\n");
        return -1;
    }

    image->sizeX = cv_image.cols;
    image->sizeY = cv_image.rows;

    GLubyte *rgba = (GLubyte *)malloc(image->sizeX * image->sizeY * 4);

    long j = 0;

    for (long y = 0; y < image->sizeY; y++)
        for (long x = 0; x < image->sizeX; x++)
        {
            rgba[j]     = cv_image.at<cv::Vec3b>(y,x)[0];
            rgba[j + 1] = cv_image.at<cv::Vec3b>(y,x)[1];
            rgba[j + 2] = cv_image.at<cv::Vec3b>(y,x)[2];
            rgba[j + 3] = 255;
            j += 4;
        }

    image->data = rgba;
    return 1;
}

Image *loadTexture(const char *image_path, bool detect = true)
{

    Image *img;

    // allocate space for texture

    img = (Image *)malloc(sizeof(Image));

    if (img == NULL)
    {

        printf("Error allocating space for image");

        exit(0);
    }

    if (!ImageLoad(image_path, img, detect))
    {
        printf("Unable to load the BMP image\n");
        exit(1);
    }

    return img;
}

/////////////////////////

long count = 0;
class Vec3 // a minimal vector class of 3 floats and overloaded math operators
{
  public:
    float f[3];

    Vec3(float x, float y, float z)
    {
        f[0] = x;
        f[1] = y;
        f[2] = z;
    }

    Vec3() {}

    float length()
    {
        return sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    }

    Vec3 normalized()
    {
        float l = length();
        return Vec3(f[0] / l, f[1] / l, f[2] / l);
    }

    void operator+=(const Vec3 &v)
    {
        f[0] += v.f[0];
        f[1] += v.f[1];
        f[2] += v.f[2];
    }

    Vec3 operator/(const float &a)
    {
        return Vec3(f[0] / a, f[1] / a, f[2] / a);
    }

    Vec3 operator-(const Vec3 &v)
    {
        return Vec3(f[0] - v.f[0], f[1] - v.f[1], f[2] - v.f[2]);
    }

    Vec3 operator+(const Vec3 &v)
    {
        return Vec3(f[0] + v.f[0], f[1] + v.f[1], f[2] + v.f[2]);
    }

    Vec3 operator*(const float &a)
    {
        return Vec3(f[0] * a, f[1] * a, f[2] * a);
    }

    Vec3 operator-()
    {
        return Vec3(-f[0], -f[1], -f[2]);
    }

    Vec3 cross(const Vec3 &v)
    {
        return Vec3(f[1] * v.f[2] - f[2] * v.f[1], f[2] * v.f[0] - f[0] * v.f[2], f[0] * v.f[1] - f[1] * v.f[0]);
    }

    float dot(const Vec3 &v)
    {
        return f[0] * v.f[0] + f[1] * v.f[1] + f[2] * v.f[2];
    }
    float angle(Vec3 &v)
    {
        float angle = acos(this->dot(v)/(this->length()*v.length()));
        return angle * 180.0 / M_PI;
    }
};

/* The particle class represents a particle of mass that can move around in 3D space*/
class Particle
{
  private:
    bool movable;            // can the particle move or not ? used to pin parts of the cloth
    float mass;              // the mass of the particle (is always 1 in this example)
    Vec3 pos;                // the current position of the particle in 3D space
    Vec3 old_pos;            // the position of the particle in the previous time step, used as part of the verlet numerical integration scheme
    Vec3 acceleration;       // a vector representing the current acceleration of the particle
    Vec3 accumulated_normal; // an accumulated normal (i.e. non normalized), used for OpenGL soft shading

  public:
    Particle(Vec3 pos) : pos(pos), old_pos(pos), acceleration(Vec3(0, 0, 0)), mass(1.0), movable(true), accumulated_normal(Vec3(0, 0, 0)) {}
    Particle() {}
    double harrisScore;
    void addForce(Vec3 f)
    {
        acceleration += f / mass;
    }

    /* This is one of the important methods, where the time is progressed a single step size (TIME_STEPSIZE)
	   The method is called by Cloth.time_step()
	   Given the equation "force = mass * acceleration" the next position is found through verlet integration*/
    void timeStep()
    {
        if (movable)
        {
            Vec3 temp = pos;
            pos = pos + (pos - old_pos) * (1.0 - DAMPING) + acceleration * TIME_STEPSIZE2;
            old_pos = temp;
            acceleration = Vec3(0, 0, 0); // acceleration is reset since it HAS been translated into a change in position (and implicitely into velocity)
        }
    }

    Vec3 &getPos() { return pos; }

    void resetAcceleration() { acceleration = Vec3(0, 0, 0); }

    void offsetPos(const Vec3 v)
    {
        if (movable)
            pos += v;
    }

    void makeUnmovable() { movable = false; }

    void addToNormal(Vec3 normal)
    {
        accumulated_normal += normal.normalized();
    }

    Vec3 &getNormal() { return accumulated_normal; } // notice, the normal is not unit length

    void resetNormal() { accumulated_normal = Vec3(0, 0, 0); }
};

class Constraint
{
  private:
    float rest_distance; // the length between particle p1 and p2 in rest configuration

  public:
    Particle *p1, *p2; // the two particles that are connected through this constraint

    Constraint(Particle *p1, Particle *p2) : p1(p1), p2(p2)
    {
        Vec3 vec = p1->getPos() - p2->getPos();
        rest_distance = vec.length();
    }

    /* This is one of the important methods, where a single constraint between two particles p1 and p2 is solved
	the method is called by Cloth.time_step() many times per frame*/
    void satisfyConstraint()
    {
        Vec3 p1_to_p2 = p2->getPos() - p1->getPos();                               // vector from p1 to p2
        float current_distance = p1_to_p2.length();                                // current distance between p1 and p2
        Vec3 correctionVector = p1_to_p2 * (1 - rest_distance / current_distance); // The offset vector that could moves p1 into a distance of rest_distance to p2
        Vec3 correctionVectorHalf = correctionVector * 0.5;                        // Lets make it half that length, so that we can move BOTH p1 and p2.
        p1->offsetPos(correctionVectorHalf);                                       // correctionVectorHalf is pointing from p1 to p2, so the length should move p1 half the length needed to satisfy the constraint.
        p2->offsetPos(-correctionVectorHalf);                                      // we must move p2 the negative direction of correctionVectorHalf since it points from p2 to p1, and not p1 to p2.
    }
};

bool sortParticles(Particle p1, Particle p2)
{
    return p1.harrisScore < p2.harrisScore;
}

class Cloth
{
  private:
    int num_particles_width;  // number of particles in "width" direction
    int num_particles_height; // number of particles in "height" direction
    // total number of particles is num_particles_width*num_particles_height

    std::vector<Particle> particles;     // all particles that are part of this cloth
    std::vector<Constraint> constraints; // alle constraints between particles as part of this cloth

    void makeConstraint(Particle *p1, Particle *p2) { constraints.push_back(Constraint(p1, p2)); }

    /* A private method used by drawShaded() and addWindForcesForTriangle() to retrieve the  
	normal vector of the triangle defined by the position of the particles p1, p2, and p3.
	The magnitude of the normal vector is equal to the area of the parallelogram defined by p1, p2 and p3
	*/
    Vec3 calcTriangleNormal(Particle *p1, Particle *p2, Particle *p3)
    {
        Vec3 pos1 = p1->getPos();
        Vec3 pos2 = p2->getPos();
        Vec3 pos3 = p3->getPos();

        Vec3 v1 = pos2 - pos1;
        Vec3 v2 = pos3 - pos1;

        return v1.cross(v2);
    }

    /* A private method used by windForce() to calcualte the wind force for a single triangle 
	defined by p1,p2,p3*/
    void addWindForcesForTriangle(Particle *p1, Particle *p2, Particle *p3, const Vec3 direction)
    {
        Vec3 normal = calcTriangleNormal(p1, p2, p3);
        Vec3 d = normal.normalized();
        Vec3 force = normal * (d.dot(direction));
        p1->addForce(force);
        p2->addForce(force);
        p3->addForce(force);
    }

    /* A private method used by drawShaded(), that draws a single triangle p1,p2,p3 with the parametrized texture*/
    void drawTriangle(Particle *p1, Particle *p2, Particle *p3, const Vec3 texcoord)
    {
        //Vec3 color = Vec3(0.5,0.5,0.5);
        Vec3 color(1.0, 1.0, 1.0);
        glColor3fv((GLfloat *)&color);

        Vec3 p1_norm = p1->getNormal().normalized();
        Vec3 p1_pos = p1->getPos();

        Vec3 p2_norm = p2->getNormal().normalized();
        Vec3 p2_pos = p2->getPos();

        Vec3 p3_norm = p3->getNormal().normalized();
        Vec3 p3_pos = p3->getPos();

        if (!(int)texcoord.f[2])
        {

            glTexCoord2f((texcoord.f[0] + 1.0) / (float)(num_particles_width - 1), 1.0 - (texcoord.f[1] / (float)(num_particles_height - 1))); //x+1,y
            glNormal3fv((GLfloat *)&(p1_norm));
            glVertex3fv((GLfloat *)&(p1_pos));

            glTexCoord2f((texcoord.f[0] / (float)(num_particles_width - 1)), 1.0 - (texcoord.f[1] / (float)(num_particles_height - 1))); //x,y
            glNormal3fv((GLfloat *)&(p2_norm));
            glVertex3fv((GLfloat *)&(p2_pos));

            glTexCoord2f(texcoord.f[0] / (float)(num_particles_width - 1), 1.0 - ((texcoord.f[1] + 1.0) / (float)(num_particles_height - 1))); //x,y+1
            glNormal3fv((GLfloat *)&(p3_norm));
            glVertex3fv((GLfloat *)&(p3_pos));
        }
        else
        {
            glTexCoord2f((texcoord.f[0] + 1.0) / (float)(num_particles_width - 1), 1.0 - ((texcoord.f[1] + 1.0) / (float)(num_particles_height - 1)));
            glNormal3fv((GLfloat *)&(p1_norm));
            glVertex3fv((GLfloat *)&(p1_pos));

            glTexCoord2f((texcoord.f[0] + 1.0) / (float)(num_particles_width - 1), 1.0 - (texcoord.f[1] / (float)(num_particles_height - 1)));
            glNormal3fv((GLfloat *)&(p2_norm));
            glVertex3fv((GLfloat *)&(p2_pos));

            glTexCoord2f(texcoord.f[0] / (float)(num_particles_width - 1), 1.0 - ((texcoord.f[1] + 1.0) / (float)(num_particles_height - 1)));
            glNormal3fv((GLfloat *)&(p3_norm));
            glVertex3fv((GLfloat *)&(p3_pos));
        }

        //vec4 color = texture2D(tex, gl_TexCoord[0].st) * gl_Color;
        //gl_FragColor = color;
    }

  public:
    /* This is a important constructor for the entire system of particles and constraints*/

    Particle *getParticle(int x, int y) { return &particles[y * num_particles_width + x]; }
    int getParticlesWidth() { return num_particles_width; }
    int getParticlesHeight() { return num_particles_height; }
    float percentile_threshold;

    void calcParticlesHarrisScores(Image *img)
    {
        percentile_threshold = 0;

        for (int y = 0; y < num_particles_height; y++)
            for (int x = 0; x < num_particles_width; x++)
            {
                int x_img = x * (img->sizeX / (float)num_particles_width);
                int y_img = y * (img->sizeY / (float)num_particles_height);

                getParticle(x, y)->harrisScore = getHarrisScore(getCvMatAt(img, y_img, x_img));
            }

        std::vector<Particle> sorted_particles = particles;

        std::sort(sorted_particles.begin(), sorted_particles.end(), sortParticles);

        //for(int i=0; i < 100; i++)
        //	std::cout << sorted_particles[i].harrisScore << std::endl;

        //getchar();

        percentile_threshold = sorted_particles[(int)sorted_particles.size() * 0.85].harrisScore;
    }

    Cloth(float width, float height, int num_particles_width, int num_particles_height) : num_particles_width(num_particles_width), num_particles_height(num_particles_height)
    {
        particles.resize(num_particles_width * num_particles_height); //I am essentially using this vector as an array with room for num_particles_width*num_particles_height particles

        // creating particles in a grid of particles from (0,0,0) to (width,-height,0)
        for (int x = 0; x < num_particles_width; x++)
        {
            for (int y = 0; y < num_particles_height; y++)
            {
                Vec3 pos = Vec3(width * (x / (float)num_particles_width),
                                -height * (y / (float)num_particles_height),
                                0);
                particles[y * num_particles_width + x] = Particle(pos); // insert particle in column x at y'th row
            }
        }

        // Connecting immediate neighbor particles with constraints (distance 1 and sqrt(2) in the grid)
        for (int x = 0; x < num_particles_width; x++)
        {
            for (int y = 0; y < num_particles_height; y++)
            {
                if (x < num_particles_width - 1)
                    makeConstraint(getParticle(x, y), getParticle(x + 1, y));
                if (y < num_particles_height - 1)
                    makeConstraint(getParticle(x, y), getParticle(x, y + 1));
                if (x < num_particles_width - 1 && y < num_particles_height - 1)
                    makeConstraint(getParticle(x, y), getParticle(x + 1, y + 1));
                if (x < num_particles_width - 1 && y < num_particles_height - 1)
                    makeConstraint(getParticle(x + 1, y), getParticle(x, y + 1));
            }
        }

        // Connecting secondary neighbors with constraints (distance 2 and sqrt(4) in the grid)
        for (int x = 0; x < num_particles_width; x++)
        {
            for (int y = 0; y < num_particles_height; y++)
            {
                if (x < num_particles_width - 2)
                    makeConstraint(getParticle(x, y), getParticle(x + 2, y));
                if (y < num_particles_height - 2)
                    makeConstraint(getParticle(x, y), getParticle(x, y + 2));
                if (x < num_particles_width - 2 && y < num_particles_height - 2)
                    makeConstraint(getParticle(x, y), getParticle(x + 2, y + 2));
                if (x < num_particles_width - 2 && y < num_particles_height - 2)
                    makeConstraint(getParticle(x + 2, y), getParticle(x, y + 2));
            }
        }

        // making the upper left most three and right most three particles unmovable
        for (int i = 0; i < 3; i++)
        {
            getParticle(0 + i, 0)->offsetPos(Vec3(randRange(0.2,0.6), 0.0, 0.0)); // moving the particle a bit towards the center, to make it hang more natural - because I like it ;)
            getParticle(0 + i, 0)->makeUnmovable();

            getParticle(num_particles_width - 1 - i, 0)->offsetPos(Vec3(-randRange(0.2,0.6), 0.0, 0.0)); // moving the particle a bit towards the center, to make it hang more natural - because I like it ;)
            getParticle(num_particles_width - 1 - i, 0)->makeUnmovable();
        }

        gridmat = cv::Mat(num_particles_height, num_particles_width, CV_32FC4);
    }

    /* drawing the cloth as a smooth shaded (and colored according to column) OpenGL triangular mesh
	Called from the display() method
	The cloth is seen as consisting of triangles for four particles in the grid as follows:

	(x,y)   *--* (x+1,y)
	        | /|
	        |/ |
	(x,y+1) *--* (x+1,y+1)

	*/
    void drawShaded()
    {

        // reset normals (which where written to last frame)
        std::vector<Particle>::iterator particle;
        for (particle = particles.begin(); particle != particles.end(); particle++)
        {
            (*particle).resetNormal();
        }

        //create smooth per particle normals by adding up all the (hard) triangle normals that each particle is part of
        for (int x = 0; x < num_particles_width - 1; x++)
        {
            for (int y = 0; y < num_particles_height - 1; y++)
            {
                Vec3 normal = calcTriangleNormal(getParticle(x + 1, y), getParticle(x, y), getParticle(x, y + 1));
                getParticle(x + 1, y)->addToNormal(normal);
                getParticle(x, y)->addToNormal(normal);
                getParticle(x, y + 1)->addToNormal(normal);

                normal = calcTriangleNormal(getParticle(x + 1, y + 1), getParticle(x + 1, y), getParticle(x, y + 1));
                getParticle(x + 1, y + 1)->addToNormal(normal);
                getParticle(x + 1, y)->addToNormal(normal);
                getParticle(x, y + 1)->addToNormal(normal);
            }
        }

#if true
        glEnable(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, texture[0]);

        glBegin(GL_TRIANGLES);
        for (int x = 0; x < num_particles_width - 1; x++)
        {
            for (int y = 0; y < num_particles_height - 1; y++)
            {
                drawTriangle(getParticle(x + 1, y), getParticle(x, y), getParticle(x, y + 1), Vec3(x, y, 0));
                drawTriangle(getParticle(x + 1, y + 1), getParticle(x + 1, y), getParticle(x, y + 1), Vec3(x, y, 1));
            }
        }

        glEnd();

        glDisable(GL_TEXTURE_2D);

#endif
    }

    /* this is an important methods where the time is progressed one time step for the entire cloth.
	This includes calling satisfyConstraint() for every constraint, and calling timeStep() for all particles
	*/
    void timeStep()
    {
        std::vector<Constraint>::iterator constraint;
        for (int i = 0; i < CONSTRAINT_ITERATIONS; i++) // iterate over all constraints several times
        {
            for (constraint = constraints.begin(); constraint != constraints.end(); constraint++)
            {
                (*constraint).satisfyConstraint(); // satisfy constraint.
            }
        }

        std::vector<Particle>::iterator particle;
        for (particle = particles.begin(); particle != particles.end(); particle++)
        {
            (*particle).timeStep(); // calculate the position of each particle at the next time step.
        }
    }

    /* used to add gravity (or any other arbitrary vector) to all particles*/
    void addForce(const Vec3 direction)
    {
        std::vector<Particle>::iterator particle;
        for (particle = particles.begin(); particle != particles.end(); particle++)
        {
            (*particle).addForce(direction); // add the forces to each particle
        }
    }

    /* used to add wind forces to all particles, is added for each triangle since the final force is proportional to the triangle area as seen from the wind direction*/
    void windForce(const Vec3 direction)
    {
        if (global_time > 40)
        {
            for (int x = 0; x < num_particles_width - 1; x++)
            {
                for (int y = 0; y < num_particles_height - 1; y++)
                {
                    addWindForcesForTriangle(getParticle(x + 1, y), getParticle(x, y), getParticle(x, y + 1), direction);
                    addWindForcesForTriangle(getParticle(x + 1, y + 1), getParticle(x + 1, y), getParticle(x, y + 1), direction);
                }
            }
        }
    }

    /* used to detect and resolve the collision of the cloth with the ball.
	This is based on a very simples scheme where the position of each particle is simply compared to the sphere and corrected.
	This also means that the sphere can "slip through" if the ball is small enough compared to the distance in the grid bewteen particles
	*/
    void ballCollision(const Vec3 center, const float radius)
    {
        std::vector<Particle>::iterator particle;
        for (particle = particles.begin(); particle != particles.end(); particle++)
        {
            Vec3 v = (*particle).getPos() - center;
            float l = v.length();
            if (v.length() < radius) // if the particle is inside the ball
            {
                (*particle).offsetPos(v.normalized() * (radius - l)); // project the particle to the surface of the ball
            }
        }
    }

    void doFrame()
    {
    }
};

/***** Above are definition of classes; Vec3, Particle, Constraint, and Cloth *****/

// Just below are three global variables holding the actual animated stuff; Cloth and Ball
Cloth cloth1(13.3, 10, 80, 60); //55,45); // one Cloth object of the Cloth class //61,48
Vec3 ball_pos(7, -5, 0);        // the center of our one ball
float ball_radius = 2;          // the radius of our one ball

/***** Below are functions Init(), display(), reshape(), keyboard(), arrow_keys(), main() *****/

/* This is where all the standard Glut/OpenGL stuff is, and where the methods of Cloth are called; 
addForce(), windForce(), timeStep(), ballCollision(), and drawShaded()*/

void init(std::string texture_path) //(GLvoid)
{

    glClearColor(0.2f, 0.2f, 0.4f, 0.5f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_COLOR_MATERIAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat lightPos[4] = {-1.0, 1.0, 0.5, 0.0};
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat *)&lightPos);

    glEnable(GL_LIGHT1);

    float sort, global_offset; 

    sort = randRange(0,1);
    //printf("Sort: %f \n", sort);

    if(sort < 0.3)
        global_offset = randRange(0.7, 0.8);
    else
    {
        global_offset = randRange(0.1, 0.4);
        //printf("Global offset: %f \n", global_offset);
    }

    GLfloat lightAmbient1[4] =  {randRange(0,0.2) + global_offset,
                                 randRange(0,0.2) + global_offset,
                                 randRange(0,0.2) + global_offset, 
                                 1.0};

    GLfloat lightPos1[4] = {1.0, 0.0, -0.2, 0.0};
    GLfloat lightDiffuse1[4] = {0.5, 0.5, 0.3, 1.0};

    glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat *)&lightPos1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat *)&lightAmbient1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat *)&lightDiffuse1);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    ////////////////////////texture part

    image1 = loadTexture(texture_path.c_str());
    if(background != "none")
        img_bg = loadTexture(background.c_str(), false);

    if (image1 == NULL || background != "none" && img_bg == NULL)
    {

        printf("Image was not returned from loadTexture\n");

        exit(0);
    }

    cloth1.calcParticlesHarrisScores(image1);
    //getCvMatAt(image1,150,110); getchar();
    //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //makeCheckImage();

    // Create Texture

    glGenTextures(3, texture);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image1->sizeX, image1->sizeY, 0, GL_BGRA, GL_UNSIGNED_BYTE, image1->data);

    if(background != "none")
    {
        glBindTexture(GL_TEXTURE_2D, texture[1]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img_bg->sizeX, img_bg->sizeY, 0, GL_BGRA, GL_UNSIGNED_BYTE, img_bg->data);
    }

    /*unsigned char pixels[] = {
		255,255,255,255,   0,0,0,255,
		255,255,255,255,    0,0,0,255

	};
	*/
    //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    int *myint;
    //glGetIntegerv(GL_UNPACK_ALIGNMENT,myint);
    //printf ("%d\n",myint[0]);

    //glPixelStorei(GL_PACK_ALIGNMENT,2);

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    //depth map texture
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); // GL_DECAL);
    /*

    glBindTexture(GL_TEXTURE_2D, texture[1]);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,GL_MODULATE);

    glTexImage2D(GL_TEXTURE_2D, 0, 3, checkImageWidth,
	
    checkImageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE,&checkImage[0][0][0]);

    //glEnable(GL_TEXTURE_2D);
    */
    glShadeModel(GL_SMOOTH);
}

float ball_time = 0; // counter for used to calculate the z position of the ball below

//////////////////////// depth map code//////////

void DrawImage(void)
{

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, glutGet(GLUT_WINDOW_WIDTH), 0.0, glutGet(GLUT_WINDOW_HEIGHT), -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glLoadIdentity();
    glDisable(GL_LIGHTING);

    glColor3f(1, 1, 1);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture[2]);

    // Draw a textured quad
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, 1);
    glVertex3f(0, Height, 0);
    glTexCoord2f(1, 1);
    glVertex3f(Width, Height, 0);
    glTexCoord2f(1, 0);
    glVertex3f(Width, 0, 0);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
}

void DrawPoint(int cx, int cy, int r, char *c, GLubyte *RGBData)
{

    //cy = Height - cy;
    if (cx - r >= 0 && cx + r < Width && cy - r >= 0 && cy + r < Height)
    {

        for (int i = -r; i < r; i++)
            for (int j = -r; j < r; j++)
            {
                RGBData[Width * 3 * (cy + i) + 3 * (cx + j)] = c[0];
                RGBData[Width * 3 * (cy + i) + 3 * (cx + j) + 1] = c[1];
                RGBData[Width * 3 * (cy + i) + 3 * (cx + j) + 2] = c[2];
            }
    }
}

float get_z_coord(float linearized_z)
{
    float zNear = 1.0;
    float zFar = 40.0;

    //return linearized_z;
    return (linearized_z * (zFar - zNear)) + zNear;
}

void save_ply(Eigen::Matrix<Vector6f, Eigen::Dynamic, Eigen::Dynamic> pointcloud)
{
    static char ply_header[] =
        "ply\n"
        "format ascii 1.0\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "element vertex %ld\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar diffuse_red\n"
        "property uchar diffuse_green\n"
        "property uchar diffuse_blue\n"
        "end_header\n";
    long num_points_out = pointcloud.rows() * pointcloud.cols();

    FILE *f = fopen("test.ply", "w");

    /* Print the ply header */
    fprintf(f, ply_header, num_points_out);

    /* X Y Z R G B for each line*/

    for (int y = 0; y < pointcloud.rows(); y++)
        for (int x = 0; x < pointcloud.cols(); x++)
            fprintf(f, "%.8f %.8f %.8f %d %d %d\n", pointcloud(y, x)(0), pointcloud(y, x)(1), pointcloud(y, x)(2), (int)(pointcloud(y, x)(3) * 255), (int)(pointcloud(y, x)(4) * 255), (int)(pointcloud(y, x)(5) * 255));

    fclose(f);
}

cv::Mat getParticleGridMat()
{
    cv::Mat gridmat(cloth1.getParticlesHeight(), cloth1.getParticlesWidth(), CV_32FC4);
}

void saveCvPng(GLubyte *RGBData, GLfloat *depth_img, std::string out_dir)
{
    cv::Mat rgb(Height,Width, CV_8UC3);
    cv::Mat depth(Height,Width, CV_16UC1);


    char buf[256];

    for (int y = 0; y < Height; y++)
        for (int x = 0; x < Width; x++)
        {
            float z = get_z_coord(depth_img[y * Width + x]) * 1000;
            char r, g, b;
            r = RGBData[Width * 3 * y + 3 * x];
            g = RGBData[Width * 3 * y + 3 * x + 1];
            b = RGBData[Width * 3 * y + 3 * x + 2];

            rgb.at<cv::Vec3b>(Height - 1 - y,x)[0] = b;  rgb.at<cv::Vec3b>(Height - 1 - y,x)[1] = g;  rgb.at<cv::Vec3b>(Height - 1 - y,x)[2] = r;  
            depth.at<uint16_t>(Height - 1 - y,x) = (uint16_t)z;
        }

    if(save_count == 0 )
    {
        sprintf(buf, "cloud_master");

        cv::Mat K = (cv::Mat_<double>(3,3) <<   0.5*Height, 0, Width / 2.,
                                        0, 0.5*Height, Height / 2.,
                                        0, 0, 1);
        cv::FileStorage fs(out_dir + "/intrinsics.xml", cv::FileStorage::WRITE);
        fs << "intrinsics" << K;
        fs.release(); 
    }
    else
        sprintf(buf, "cloud_%d", save_count);

    
    cv::FileStorage gfs(out_dir + "/" + buf + "-grid.xml", cv::FileStorage::WRITE);
    gfs << "grid" << gridmat;
    gfs.release();
    
    /** Add a little gaussian noise to rgb image **/
    cv::Mat mGaussian_noise = cv::Mat(rgb.size(),CV_32FC3);
    cv::Mat result;
    rgb.convertTo(result, CV_32FC3);
    cv::randn(mGaussian_noise,0,3);
    result += mGaussian_noise;
    result.setTo(0, result < 0);
    result.setTo(255, result > 255);
    result.convertTo(rgb, CV_8UC3);

    cv::imwrite(out_dir + "/" + buf + "-rgb.png", rgb);
    cv::imwrite(out_dir + "/" + buf + "-depth.png", depth);
    //printf("Saved depth and png (%d) in: %s\n", save_count, out_dir.c_str());

    save_count++;

    if(save_count == max_frames)
    {
        if(verbose == "true")
            printf("Finishing the simulation (achieved max frames).\n");
        
        exit(0);
    }

}


Eigen::Matrix<Vector6f, Eigen::Dynamic, Eigen::Dynamic> convert_to_xyzrgb(GLubyte *RGBData, GLfloat *depth_img)
{

    Eigen::Matrix<Vector6f, Eigen::Dynamic, Eigen::Dynamic> pointcloud(Height, Width);
    Eigen::Matrix3f K; // Intrinsic matrix for 90.0 FoV WidthxHeight camera

    K << 240, 0, 320,
        0, 240, 240,
        0, 0, 1;

    Eigen::Vector3f pt2d; // 2D point

    for (int y = 0; y < Height; y++)
        for (int x = 0; x < Width; x++)
        {
            float z = get_z_coord(depth_img[y * Width + x]);
            char r, g, b;
            r = RGBData[Width * 3 * y + 3 * x];
            g = RGBData[Width * 3 * y + 3 * x + 1];
            b = RGBData[Width * 3 * y + 3 * x + 2];

            pt2d << x, Height - 1 - y, 1.0; // invert y axis to standard in pcl, meshlab, etc...
            Eigen::VectorXf pt2d_norm = K.inverse() * pt2d;
            Vector6f point;
            point << pt2d_norm(0) * z, pt2d_norm(1) * z, z, r / 255.0, g / 255.0, b / 255.0;
            pointcloud(Height - 1 - y, x) = point; // invert y axis to standard in pcl, meshlab, etc...
                                                //printf("%d %d \n",x,y);
        }

    return pointcloud;
}

bool projectPointGl(GLubyte *data, GLfloat *zdata, bool save)
{
    //std::vector<GLfloat> P(16);
    Eigen::Matrix4f P; //Projection Matrix
    Eigen::Vector4f X; // 3D point
    Eigen::Matrix3f K; // Intrinsic matrix for 90.0 FoV WidthxHeight camera

    K << 0.5*Height, 0, Width / 2.,
        0, 0.5*Height,  Height/ 2.,
        0, 0, 1;

    glGetFloatv(GL_MODELVIEW_MATRIX, P.data());

    FILE *f;

    if (save)
    {
        char buffer[128];

        if (save_count == 0)
            sprintf(buffer, "%s/cloud_master.csv", out_dir.c_str());
        else
            sprintf(buffer, "%s/cloud_%d.csv", out_dir.c_str(), save_count);


        f = fopen(buffer, "w");
        fprintf(f, "id,x,y,valid\n");
    }

    int kp_id = 0;
    int num_valid = 0;
    cv::Vec4f vf;

    //fill gridmat 
    for(int y =0; y < cloth1.getParticlesHeight(); y++)
        for(int x =0; x < cloth1.getParticlesWidth(); x++)
        {
            Vec3 kp_3d = cloth1.getParticle(x,y)->getPos();
            X << kp_3d.f[0],kp_3d.f[1], kp_3d.f[2], 1.0;
            Eigen::MatrixXf p_c = P * X; // Transform point to camera coordinate system
            p_c(2) *= -1;

            Eigen::Vector3f pj;
            pj << p_c(0) / p_c(2), p_c(1) / p_c(2), 1.0; // Perspective projection
            Eigen::VectorXf px;
            px = K * pj; 
              
            vf[0] = p_c(0); vf[1] = p_c(1); vf[2] = p_c(2);
            vf[3] = 0;


            if (px(0)-5 >= 0 && px(0)+5 < Width && px(1)-5 >= 0 && px(1)+5 < Height && p_c(2) > 1.0)
            {
                float z_est = get_z_coord(zdata[Width * (int)(px(1)) + (int)(px(0))]); 
                if(p_c(2) <= z_est + 0.05)
                    vf[3] = 1;
            }

            gridmat.at<cv::Vec4f>(y,x) = vf;
        }


    for(int i=0; i < keypoints.size(); i++)
    {
        float x = (keypoints[i].pt.x / (image1->sizeX -1)) * (cloth1.getParticlesWidth() -1);
        float y = (keypoints[i].pt.y / (image1->sizeY -1)) * (cloth1.getParticlesHeight() -1);

        //printf("kp.p = %.2f %.2f\n", x, y); getchar();

        float rx, ry;
        rx = x - (int)x;
        ry = y - (int)y;

        Vec3 a, b, c, ab, ac;

        if(rx + ry <= 1.0)
        {
            a = cloth1.getParticle((int)x, (int)y)->getPos();
            b = cloth1.getParticle((int)x, (int)y+1)->getPos();
            c = cloth1.getParticle((int)x+1, (int)y)->getPos();
        }
        else
        {
            a = cloth1.getParticle((int)x+1, (int)y+1)->getPos();
            c = cloth1.getParticle((int)x, (int)y+1)->getPos();
            b = cloth1.getParticle((int)x+1, (int)y)->getPos(); 

            rx = 1.0 - rx;
            ry = 1.0 - ry;               
        }

        ab = b - a;
        ac = c - a;
        Vec3 kp_3d = a + (ac * rx + ab* ry);

        X << kp_3d.f[0],kp_3d.f[1], kp_3d.f[2], 1.0;

        Eigen::MatrixXf p_c = P * X; // Transform point to camera coordinate system

        p_c(2) *= -1;

        //printf("z_ground_truth = %.8f\n", p_c(2));

        Eigen::Vector3f pj;
        pj << p_c(0) / p_c(2), p_c(1) / p_c(2), 1.0; // Perspective projection
        Eigen::VectorXf px;
        px = K * pj;

        //zdata[Width* (int)(px(1)) + (int)(px(0))]	 = 0.1;

        //check Z coordinate
        float z_est = 0;
        px(0)= (int)(px(0) + 0.5);
        px(1)= (int)(px(1) + 0.5);

        if (px(0)-5 >= 0 && px(0)+5 < Width && px(1)-5 >= 0 && px(1)+5 < Height && p_c(2) > 1.0)
        {
            float z_est = get_z_coord(zdata[Width * (int)(px(1)) + (int)(px(0))]);

            char c1[] = {255, 0, 0};
            char c2[] = {0, 0, 255};

            if (p_c(2) <= z_est + 0.05)
            {
                if(!save)
                    DrawPoint(px(0), px(1), 1, c1, data);
                num_valid++;
                /* Print keypoint info */
                if (save)
                    fprintf(f, "%d,%.2f,%.2f,%d\n", kp_id++, px(0), Height - px(1), 1);
            }
            else //occluded point
            {
                if(!save)
                    DrawPoint(px(0), px(1), 1, c2, data);
                /* Print keypoint info */
                if (save)
                    fprintf(f, "%d,%.2f,%.2f,%d\n", kp_id++, px(0), Height - px(1), 0);
            }

            //printf("z_global = %.8f\n",X(2));
            //printf("z_estimated = %.8f\n\n", z_est);
            //printf ("z_linearized = %f\n\n",zdata[(int)( px(1)*Width + px(0) )]);
        }
        else if (save)
            fprintf(f, "%d,%.2f,%.2f,%d\n", kp_id++, px(0), Height - px(1), 0);

    }
    
    

    if (save)
        fclose(f);

    if(num_valid/ (double)keypoints.size() > 0.6 )
        return true;

    return false;
    // std::cout << "P = [" << P << "]" << std::endl;
    //std::cout << "px = [" << px << "]" << std::endl;
 }

float get_normalized_z(float depth)
{
    float zFar = 40.0;
    float zNear = 1.0;

    float z_b = depth;
    float z_n = 2.0 * z_b - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear)); // Actual Z coordinates

    return (z_e - zNear) / (zFar - zNear);
}

void CopyImageBuffers(GLuint texId, int imageWidth, int imageHeight)
{
    int x = 0;
    int y = 0;

    float zFar = 40.0;
    float zNear = 1.0;

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texId);
    glReadBuffer(GL_BACK); // Ensure we are reading from the back buffer.

    GLfloat *zData = new GLfloat[Width * Height];
    GLubyte *RGBData = new GLubyte[Width * Height * 3];

    GLubyte *RGBFull = new GLubyte[Width * 2 * Height * 3];

    glReadPixels(x, y, Width, Height, GL_DEPTH_COMPONENT, GL_FLOAT, zData);
    glReadPixels(x, y, Width, Height, GL_RGB, GL_UNSIGNED_BYTE, RGBData);

    for (long i = 0; i < Width * Height; i++)
    {
        zData[i] = get_normalized_z(zData[i]); //(2.0 * zNear) / (zFar + zNear - zData[i] * (zFar-zNear));
    }

    if (save_result)
    {
        //save_ply(convert_to_xyzrgb(RGBData, zData));
        //convert_to_mesh(convert_to_xyzrgb(RGBData, zData), out_dir);
        if(projectPointGl(RGBData, zData, true))
        {
            saveCvPng(RGBData, zData, out_dir);
            if(verbose == "true")
                std::cout << " * Saved mesh and png image in global time: " << global_time << std::endl;
        }

        save_result = false;
    }

    projectPointGl(RGBData, zData, false);

    for (long y = 0; y < Height; y++)
        for (long x = 0; x < Width * 2; x++)
            for (long p = 0; p < 3; p++)
            {
                if (x < Width)
                    RGBFull[Width * 2 * 3 * y + 3 * x + p] = RGBData[Width * 3 * y + 3 * x + p];
                else
                    RGBFull[Width * 2 * 3 * y + 3 * x + p] = (unsigned char)(zData[Width * y + x - Width] * 255);
            }

    //if(dmap)
    //glTexImage2D(GL_TEXTURE_2D, 0, 1, Width, Height, 0, GL_RED, GL_FLOAT, zData);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, Width, Height, 0, GL_RGB, GL_UNSIGNED_BYTE, RGBData);
    //else

    //glTexImage2D(GL_TEXTURE_2D, 0, 3, 1280, Height, 0, GL_RGB, GL_UNSIGNED_BYTE, RGBFull);

    delete zData;
    delete RGBData;
    delete RGBFull;

    //std::cout << _save_result << std::endl;

    glDisable(GL_TEXTURE_2D);
    //glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, x, y, imageWidth, imageHeight, 0);
}


void camera(void)
{

    //compute camera look
    Vec3 ray_x =  Vec3(w_half, 0, 0) - Vec3(-xpos + xpos_off, 0,-zpos);
    Vec3 ray_y = Vec3(0, h_half, 0) - Vec3(0, -ypos + ypos_off, -zpos);
    Vec3 unit_z = Vec3(0,0,-1);
    Vec3 yn = Vec3(0,1,0);
    Vec3 xn = Vec3(1,0,0);

    //compute x angle
    float x_angle = ray_x.angle(unit_z);
    Vec3 n = ray_x.cross(unit_z);
    if(n.dot(yn) < 0)
        x_angle = -x_angle;

    //compute y angle
    float y_angle = ray_y.angle(unit_z);
    n = ray_y.cross(unit_z);
    if(n.dot(xn) < 0)
        y_angle = -y_angle;

    //printf("ANGLES: %.3f   %.3f   \n", x_angle, y_angle);
    //printf("ray = %.3f %.3f %.3f \n", ray_x.f[0], ray_x.f[1], ray_x.f[2]);
    //y_angle = 0; x_angle = 0;
    glRotatef(xrot - y_angle, 1.0, 0.0, 0.0); //rotate our camera on the y-axis (up and down)
    glRotatef(yrot - x_angle, 0.0, 1.0, 0.0); //rotate our camera on the  x-axis (left right)
    glRotatef(zrot, 0.0, 0.0, 1.0); //rotate in-plane
    glTranslated(xpos + xpos_off, ypos + ypos_off, zpos); //translate the screen to the position of our camera
}

/*float randRange(float min, float max){
    return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}*/



///////////////////////////////////////////////
/* display method called each frame*/
void display(void)
{
    if(variation_interval > 0 && global_time % variation_interval == 0){
        // printf("\n-------------RAND-------------\n");
        if(fx_var > 0)
            _fx = randRange(-fx_var, fx_var) + fx;
        
        if(fy_var > 0)
            _fy = randRange(-fy_var, fy_var) + fy;
        
        if(fz_var > 0)
            _fz = randRange(-fz_var, fz_var) + fz;

        if(light_variation > 0){
            int varx = randRange(-light_variation*3, light_variation*13);
            int vary = randRange(-light_variation*13, light_variation*3);
            int varz = randRange(-light_variation*5, light_variation*5);

            GLfloat lightPos[4] = {1.0 + varx, 1.0 + vary, 0.5 + varz, 0.0};
            glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat *)&lightPos);

            varx = randRange(-light_variation*3, light_variation*13);
            vary = randRange(-light_variation*13, light_variation*3);
            varz = randRange(-light_variation*5, light_variation*5);

            GLfloat lightPos1[4]     = {1.0 + varx, 0.0 + vary, -0.2 + varz, 0.0};
            glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat *)&lightPos1);

        }
    }

    // calculating positions
    //glBindTexture(GL_TEXTURE_2D, 0);

    ball_time++;
    global_time++;

    if (global_time > 80)
    {   
        if(rot_increment > 0){
            zrot += rot_increment;
        }

        if(rot_var > 0)
        {
            zrot = randRange(-rot_var, rot_var);   
        }
        if(cx > 0)
            xpos_off = randRange(-cx, cx);
        if(cy > 0)
            ypos_off = randRange(-cy, cy);

        // if (zrot > max_rot)
        // {
        //     exit(0);
        // }


    }

    if (save_interval > 0 && global_time % (save_interval + (int)randRange(0,save_range+1)) == 0 && _save_result && 
        global_time > wait_time)
    {
        //std::cout << "Saving with zrot " << zrot << std::endl;
        save_result = true;
    }

    ball_pos.f[2] = cos(ball_time / 50.0) * 7; //*7

    cloth1.addForce(Vec3(0, -0.2 * gravity, 0) * TIME_STEPSIZE2);     // add gravity each frame, pointing down //0.2
    // printf("%f %f %f", _fx, _fy, _fz);
    cloth1.windForce(Vec3(_fx, _fy, _fz) * TIME_STEPSIZE2); // generate some wind each frame //0.5 , 0 , 0.2
    cloth1.timeStep();                                   // calculate the particle positions of the next frame
    // cloth1.ballCollision(ball_pos,ball_radius); // resolve collision with the ball

    // drawing

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    //glDisable(GL_LIGHTING); // drawing some smooth shaded background - because I like it ;)
        if(background != "none")
        {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, texture[1]);
        }

        // // Draw a textured quad
        // glBegin(GL_QUADS);
        // glTexCoord2f(0, 0);
        // glVertex3f(0, 0, 0);
        // glTexCoord2f(0, 1);
        // glVertex3f(0, Height, 0);
        // glTexCoord2f(1, 1);
        // glVertex3f(Width, Height, 0);
        // glTexCoord2f(1, 0);
        // glVertex3f(Width, 0, 0);
        // glEnd();


        camera(); // transform camera according to position and orientation

        glBegin(GL_POLYGON);
        if(background == "none")
            glColor3f(1.0f, 1.0f, 1.0f);
        else
            glTexCoord2f(0, 0);
        glVertex3f(-80.0f, -60.0f, -15.0f);
        if(background != "none")
            glTexCoord2f(1, 0);
        glVertex3f(80.0f, -60.0f, -15.0f);
        if(background == "none")
            glColor3f(1.0f, 1.0f, 1.0f);
        else
            glTexCoord2f(1, 1);
        glVertex3f(80.0f, 65.0f, -15.0f);
        if(background != "none")
            glTexCoord2f(0,1);
        glVertex3f(-80.0f, 65.0f, -15.0f);
        glEnd();
        //glEnable(GL_LIGHTING);

    if(background != "none")
        glDisable(GL_TEXTURE_2D);

    /*glPushMatrix();
		glTranslatef(5,8,-10);
		glutSolidSphere(1,50,50);
	glPopMatrix();
	*/

    
    //glTranslatef(-6.5,5,-7.0f); // move camera out and center on the cloth
    //glTranslatef(0,0,-10.0f);
    //glRotatef(15,0,1,0); // rotate a bit to see the cloth from the side
    cloth1.drawShaded(); // finally draw the cloth with smooth shading

    glPushMatrix();                                            // to draw the ball we use glutSolidSphere, and need to draw the sphere at the position of the ball
    glTranslatef(ball_pos.f[0], ball_pos.f[1], ball_pos.f[2]); // hence the translation of the sphere onto the ball position
    glColor3f(0.4f, 0.8f, 0.5f);
    //glutSolidSphere(ball_radius-0.1,50,50); // draw the ball, but with a slightly lower radius, otherwise we could get ugly visual artifacts of cloth penetrating the ball slightly
    glPopMatrix();

    CopyImageBuffers(texture[2], Width, Height); //handle depth buffer and RGB image

    DrawImage(); // draw the image onto the screen

    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    //printf("%d %d .... \n", h, w);

    if(h != Height || w != Width)
    {
        printf("ERROR, intrinsics became invalid because of window resizing, exiting...\n");
        exit(0);
    }

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (h == 0)
        gluPerspective(90, (float)w, 1.0, 40.0);
    else
        gluPerspective(90, (float)w / (float)h, 1.0, 40.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        exit(0);
        break;
    /*case 'f':
        save_result = true;
        break;
    */
    default:
        break;
    }

    float xforward = 0.2 * cos(yrot * (M_PI / 180.0));
    float yforward = 0.2 * sin(yrot * (M_PI / 180.0));

    float xsideward = -0.2 * sin(yrot * (M_PI / 180.0));
    float ysideward = 0.2 * cos(yrot * (M_PI / 180.0));

    if (key == 'w')
    {
        xpos -= yforward;
        zpos += xforward;
    }

    if (key == 's')
    {
        xpos += yforward;
        zpos -= xforward;
    }

    if (key == 'd')
    {
        xpos -= ysideward;
        zpos += xsideward;
    }

    if (key == 'a')
    {
        xpos += ysideward;
        zpos -= xsideward;
    }

    if (key == 'x')
    {
        ypos += 0.1;
    }
    if (key == 'c')
    {
        ypos -= 0.1;
    }

    if (key == 'z')
        zrot += 5.0;
    if (key == 'v')
        zrot -= 5.0;

    //std::cout << xpos << "," << ypos << "," << zpos << " | " << zrot << std::endl;
}

void mouseMovement(int x, int y)
{
    int diffx = x - lastx; //check the difference between the current x and the last x position
    int diffy = y - lasty; //check the difference between the current y and the last y position
    lastx = x;             //set lastx to the current x position
    lasty = y;             //set lasty to the current y position
    xrot -= (float)diffy;  //set the xrot to xrot with the addition of the difference in the y position
    yrot -= (float)diffx;  //set the xrot to yrot with the addition of the difference in the x position
}

void arrow_keys(int a_keys, int x, int y)
{
    switch (a_keys)
    {
    case GLUT_KEY_UP:
        xrot -= 5.0f;
        //glutFullScreen();
        break;
    case GLUT_KEY_DOWN:
        xrot += 5.0f;
        //glutReshapeWindow (Width, Height );
        break;
    case GLUT_KEY_LEFT:
        yrot -= 5.0f;
        break;
    case GLUT_KEY_RIGHT:
        yrot += 5.0f;
        break;
    default:
        break;
    }
}

void parseArgs(int argc, char **argv){
    char *opt;

    if(cmdOptionExists(argv, argv + argc, "-h")){
        printf("Options:\n");
        printf("Wind force \n");
        printf("[--fx <float>] [--fy <float>] [--fz <float>]\n");
        printf("[--fx_var <float>] [--fy_var <float>] [--fz_var <float>]\n");
        printf("Light variation, rotation and camera distance\n");
        printf("[--light_var <float>] [--rot <float>] [--cam_dist <float>]\n");
        printf("Camera position offset\n");
        printf("[--cx <float>] [--cy <float>]\n");
        printf("Out directory and Save flag\n");
        printf("[--out_dir <string>] [--save]\n");
        printf("Time interval to variate params and save simulation\n");
        printf("[--variation_interval <int>] [--save_interval <int>]\n");
        printf("Random seed\n");
        printf("[--seed <int>]\n");
        printf("[--max_frames <int>]\n");
        printf("[--verbose <string>]\n");
        printf("Width and Height resolution of the simulation\n");
        printf("[--width <int>]\n");
        printf("[--height <int>]\n");
        exit(0);
    }

    // Forces    
    opt = getCmdOption(argv, argv + argc, "--fx");
    fx = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--fy");
    fy = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--fz");
    fz = (opt) ? atof(opt) : 0.2;

    // Variantions
    opt = getCmdOption(argv, argv + argc, "--fx_var");
    fx_var = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--fy_var");
    fy_var = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--fz_var");
    fz_var = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--light_var");
    light_variation = (opt) ? atof(opt) : 0;
    
    opt = getCmdOption(argv, argv + argc, "--rot");
    rot_increment = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--rot_var");
    rot_var = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--cx");
    cx = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--cy");
    cy = (opt) ? atof(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--gravity");
    gravity = (opt) ? atof(opt) : 1.0;

    // Time to variate    
    opt = getCmdOption(argv, argv + argc, "--save_interval");
    save_interval = (opt) ? atoi(opt) : 0;
    opt = getCmdOption(argv, argv + argc, "--save_range");
    save_range = (opt) ? atoi(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--variation_interval");
    variation_interval = (opt) ? atoi(opt) : 0;

    // Camera distance
    opt = getCmdOption(argv, argv + argc, "--cam_dist");
    zpos = (opt) ? atof(opt) : -7;

    // Width & Height
    opt = getCmdOption(argv, argv + argc, "--width");
    Width = (opt) ? atof(opt) : 641-1;
    opt = getCmdOption(argv, argv + argc, "--height");
    Height = (opt) ? atof(opt) : 481-1;

    // Output dir to save simulation    
    opt = getCmdOption(argv, argv + argc, "--out_dir");
    out_dir = (opt) ? std::string(opt) :  std::string("result/");

    // Random seed
    opt = getCmdOption(argv, argv + argc, "--seed");
    unsigned long seed = (opt) ? (unsigned long)atoi(opt) : 0;
    mt =  std::mt19937(seed * 123456);
    cv::theRNG().state = seed * 123456;

    // Random seed
    opt = getCmdOption(argv, argv + argc, "--max_frames");
    max_frames = (opt) ? atoi(opt) : 30;

    opt = getCmdOption(argv, argv + argc, "--wait_time");
    wait_time = (opt) ? atoi(opt) : 0;

    opt = getCmdOption(argv, argv + argc, "--verbose");
    verbose = (opt) ? std::string(opt) :  std::string("true");

    opt = getCmdOption(argv, argv + argc, "--background");
    background = (opt) ? std::string(opt) :  std::string("none");

    // Flag to save simulation
    _save_result = cmdOptionExists(argv, argv + argc, "--save");
    if(_save_result){
        if(save_interval <=0 ){
            printf("Use --time to define a positive time interval to save the simularion\n");
            exit(0);
        }

        if(verbose == "true")
            printf("Outdir: %s\n", out_dir.c_str());
        //system(("mkdir -p " + out_dir).c_str() );
    }

    _fx = fx; 
    _fy = fy; 
    _fz = fz; 


    if(verbose == "true")
    {

        printf("Wind Force:           (%f, %f, %f)\n", _fx, _fy, _fz);
        printf("Force Variation:      (%f, %f, %f)\n", fx_var, fy_var, fz_var);
        printf("Light Variation:      (%f, %f)\n", -light_variation, light_variation);
        printf("Rotation Increment:    %f\n", rot_increment);
        printf("Variation Interval:    %i\n", variation_interval);
        printf("Save Interval:         %i\n", save_interval);
        printf("Save Range:            %i\n", save_range);
        printf("Wait Time:             %i\n", wait_time);
        printf("Cam Distance:          %f\n", zpos);
        printf("Random seed:           %ld*123456\n", seed);
        printf("Max frames:            %d\n", max_frames);
        printf("Width:                 %ld\n", Width);
        printf("Height:                %ld\n", Height);


    }


}

int main(int argc, char **argv)
{
    parseArgs(argc, argv);

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(Width, Height);

    glutCreateWindow("Cloth Simulation + Texturing (OpenGL) - press 'ESC' to quit");
    max_rot = 180;


    init(std::string(argv[1]));
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    //glutPassiveMotionFunc(mouseMovement); //check for mouse
    glutSpecialFunc(arrow_keys);

    glutMainLoop();
}