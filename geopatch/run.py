#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%   This file is part of https://github.com/verlab/GeoPatch_CVIU_2022
#
#    geopatch-descriptor is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geopatch-descriptor is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with geopatch-descriptor.  If not, see <http://www.gnu.org/licenses/>.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''

This script automatically runs the geopatch code and save the results with all desired datasets.
--input: .txt file containing a list of paths, where each path points to the base directory of a dataset.
--output: Output path where file results are going to be saved.

Example:
python run --input /home/user/datasets/run_all_experiments.txt --output /home/user/nonrigid/results

'''

import os, sys

nrigid_bin_folder = os.path.dirname(os.path.realpath(__file__))  +'/build/' #Set the path to the compute_descriptor executable.

import subprocess
import glob
import argparse
import re

if sys.version_info[0] == 3:
	def raw_input():
		return input()

nrigid_bin_folder = os.path.abspath(nrigid_bin_folder)+'/'

pyramid_nlevels = 0
kp_scales = [1.0] #Desired scales to test
isocurvesizes = [0.05] #Desired iso sizes
experiment_name = 'results' #Desired experiment name root folder

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]

	return dirs

CWD = './build/tmp' ; check_dir(CWD)

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input file containing a list of experiments folders"
	, required=True, default = 'lists.txt') 
	parser.add_argument("-o", "--output", help="Output path where file results are going to be saved."
	, required=True, default = '')
	parser.add_argument("-m", "--mode", help="Mode [groundtruth] matching, [patch] extraction"
	, required=True, choices = ['groundtruth', 'patch']) 
	parser.add_argument("-f", "--file", help="is a file list?"
	, action = 'store_true')  
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true')  	
	args = parser.parse_args()
	return args

def standarize_csv_names(csv_list):
	for csv in csv_list:
		csv_path = os.path.dirname(csv)
		csv_name = os.path.basename(csv)
		csv_rename = re.findall('cloud_[0-9a-zA-Z]+',csv_name)[0] + '.pcd.csv'
		if csv_name != csv_rename:
			command = 'mv ' + csv_path + '/' + csv_name + ' ' + csv_path + '/' + csv_rename
			proc = subprocess.Popen(['/bin/bash','-c',command])	
			proc.wait()
	
'''
def intensity_centroid(patch):
	acc = [0] * len(patch.shape[0])

	for d in range(len(patch.shape[0])):
		for l in range(len(patch.shape[1])):
			acc+= patch[d,l]

	return np.roll(patch, shift, axis = 1)
'''

def main():

	args = parseArg()
	args.input = os.path.abspath(args.input) 
	args.output = os.path.abspath(args.output) 

	if args.file:
		exp_list = get_dir_list(args.input)
	elif args.dir:
		exp_list = [d for d in glob.glob(args.input+'/*') if os.path.isdir(d)]
	else:
		exp_list = [args.input]

	for exp_dir in exp_list:
		datasettype_flag = ''

		if 'synthetic' in exp_dir.lower() or 'simulation' in exp_dir.lower():
			datasettype_flag = 'synthetic-'
			isocurvesizes = [1300] # 0.05
			smooth = 'false'
			scale = 1.0 
		else:
			datasettype_flag = 'realdata-'
			isocurvesizes = [75]#[100] #75
			smooth = 'true'
			scale = 0.5


		#dataset_name = os.path.basename(os.path.dirname(exp_dir))
		dataset_name = os.path.basename(exp_dir) #; print(dataset_name) ; input()
		#dataset_name = os.path.abspath(exp_dir).split('/')[-2] + '_' +  os.path.abspath(exp_dir).split('/')[-1]
		#print(dataset_name) ; raw_input()
		
		experiment_files_unfiltered = glob.glob(exp_dir + "/*-rgb.png")
		experiment_files = [os.path.basename(e).split('-rgb.png')[0] for e in experiment_files_unfiltered]

		#print( experiment_files) ; quit()
		#standarize_csv_names(glob.glob(exp_dir + "/*.csv"))
		
		master_f = ''
		target_files = []
		for exp_file in experiment_files:
			if 'master' in exp_file or 'ref' in exp_file:
				master_f = os.path.basename(exp_file)
			else:
				target_files.append(os.path.basename(exp_file))


		for kp_scale in kp_scales:
			for isocurvesize in isocurvesizes:
				
				#building command
				command = nrigid_bin_folder + './geodesic_patch -inputdir '
				command+= exp_dir
				command+= ' -refcloud ' + master_f

				for target_file in target_files:
					command+= ' -clouds ' + target_file

				command+= ' -radius ' + str(isocurvesize)
				command+= ' -smooth ' + smooth
				command+= ' -scale ' + str(scale)
				command+= ' -mode ' + args.mode


				master_f_name, _ = os.path.splitext(os.path.basename(master_f))
				#print( command) +'\n\n'
				proc = subprocess.Popen(['/bin/bash','-c','rm ' + '*.png' ], cwd = CWD) #clean old data
				proc.wait()
				
				supercommand = '{ time %s ; } 2> %s_time.txt'%(command,master_f_name)
				#print( supercommand) ; input('..')
				proc = subprocess.Popen(['/bin/bash','-c',supercommand], cwd = CWD)	
				proc.wait()

				result_dir = os.path.join(args.output,experiment_name) + '/' + dataset_name
				#print( result_dir + '\n') ;#raw_input()
				check_dir(result_dir)


				if len(master_f_name) > 5:
					command = 'mv -f ' + '*' + master_f_name + '* ' + result_dir ; #print( command + '\n\n')
					proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
					proc.wait()

				command = 'mv -f ' + '*.png ' + result_dir ; #print( command + '\n\n')
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
				proc.wait()

				command = 'mv -f ' + '*' + '.ply ' + result_dir ; #print( command + '\n\n')
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
				proc.wait()

				command = 'cp ' + '*' + '.csv ' + result_dir ; #print( command + '\n\n')
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = exp_dir)	
				proc.wait()

main()
