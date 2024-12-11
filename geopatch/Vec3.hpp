  
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

#ifndef VEC3
#define VEC3
// @Brief
// Minimalist class for 3D vector operations
class Vec3
{
	public:

	 double x, y, z;

		Vec3(double _x, double _y, double _z)
		{
			x = _x;  y = _y;  z = _z;
		}

		Vec3() {}

		double norm()
		{
			return sqrt(x * x + y * y + z * z);
		}

		Vec3 unit()
		{
			double l = norm();
			return Vec3(x/l, y/l, z/l);
		}
		
		Vec3 operator+(const Vec3 &v)
		{
			return Vec3(x + v.x,  y + v.y,  z + v.z);
		}

		Vec3 operator-(const Vec3 &v)
		{
			return Vec3(x - v.x,  y - v.y,  z - v.z);
		}
		Vec3 operator-()
		{
			return Vec3(-x, -y, -z);
		}
		bool operator==(const Vec3 &v)
		{
			static double tol = 1e-12;
			//return (x == v.x && y == v.y && z == v.z);
			return (x >= v.x - tol && x <= v.x + tol) &&
				   (y >= v.y - tol && y <= v.y + tol) &&
				   (z >= v.z - tol && z <= v.z + tol);
		}
		double dot(const Vec3 &v)
		{
			return x * v.x + y * v.y + z * v.z;
		}

		Vec3 cross(const Vec3 &v)
		{
			return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
		}

		Vec3 operator*(const double c)
		{
			return Vec3(x*c, y*c, z*c);
		}
		Vec3 operator/(const double c)
		{
			return Vec3(x/c, y/c, z/c);
		}
		bool isNull()
		{
			if(x < 1e-40 && y < 1e-40 && z < 1e-40)
				return true;
			return false;
		}
};

#endif