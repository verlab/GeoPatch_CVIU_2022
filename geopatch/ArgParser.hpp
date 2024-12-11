  
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


#ifndef _PARSER_
#define _PARSER_

#include <iostream>
#include <string>
#include <map>
#include <stdlib.h>

using namespace std;

typedef map<string, vector<string > > Dict;

class ArgParser
{
	Dict args_map;

	public:

		ArgParser(int argc, char* argv[])
		{
			try
			{
				if((argc+1)%2 !=0)
					throw 1;				
				
				/*******Default Arguments**********/
				args_map["-inputdir"] = vector<string>();
				args_map["-refcloud"] = vector<string>();
				args_map["-clouds"] = vector<string>();
				args_map["-scale"] = vector<string>();
				args_map["-smooth"] = vector<string>();
				args_map["-radius"] = vector<string>();
				args_map["-mode"] = vector<string>();


				/***********************************/

				for(size_t i=0; i< argc - 1; i++)
				{
					if(args_map.find(string(argv[i])) != args_map.end())
						args_map[string(argv[i])].push_back(argv[i+1]);
					else if(i%2!=0)
						cout<<"Warning: argument "<<argv[i]<< " not valid!"<<endl;
				}

				for(Dict::iterator it = args_map.begin(); it != args_map.end(); ++it)
					if(it->second.size() == 0)
						{throw 2;}
				

				cout<<"---------------------------------------------" << endl;
				cout<<"Using parameters:"<<endl;
				for(Dict::iterator it = args_map.begin(); it != args_map.end(); ++it)
				{
					for(int i=0; i < it->second.size(); i++)
						cout<<it->first<<" "<<it->second[i]<<endl;
				}
				cout<<"---------------------------------------------" << endl;

			}
			
			catch (int e)
			{
				if (e==1)
				{
					cout<<"Error: all parameters need to be in the format: [--flag_name parameter_value], e.g. :"<<endl;
					cout<<"./program --flag1 val1 --flag2 val2 --flag3 val3" <<endl;
					cout<< endl << ">> Usage Example: --dir /home/user --file somefile.png" << endl;
				}
				else if(e==2)
				{
					cout<<"Error: The parameters -inputdir -refcloud -scale -smooth -radius are required!"<<endl;
				}

				
				exit (EXIT_FAILURE);
			}			
		}

		Dict get_args(){return args_map;}
};

#endif