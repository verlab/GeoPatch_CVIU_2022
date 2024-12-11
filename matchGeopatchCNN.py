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


#!/usr/bin/env python
# coding: utf-8
import cv2

import os
import subprocess
import glob
import argparse
import numpy as np
import re
import multiprocessing
import time
from scipy.spatial import distance
import pdb

experiment_name = 'results'
exp_dir_target = ''

import architecture as arch

net = arch.TinyDesc('ours_shift8x3equal_circularpad_train200')

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input path containing single or several (use -d flag) PNG-CSV dataset folders"
	, required=True, default = 'lists.txt') 
	#parser.add_argument("-o", "--output", help="Output path where results will be saved."
	#, required=True, default = '.') 
	parser.add_argument("-f", "--file", help="Use file list with several input dirs instead (make sure -i points to .txt path)"
	, action = 'store_true') 
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true')  	
	args = parser.parse_args()
	return args

def correct_cadar_csv(csv):
	for line in csv:
		if line['x'] < 0 or line['y'] < 0:
			line['valid'] = 0

def load_raw_patches(filename):
	 patch_size = 32
	 raw_patches = cv2.imread(filename, 0)
	 n_images = int(raw_patches.shape[1]/patch_size)
	 return [raw_patches[:, i * patch_size : (i+1) * patch_size] for i in range(n_images)]

def toTFShape(patches):

	patches = [p for p in patches if not np.all(p == 255.0)]

	patches = np.array(patches, dtype = np.float32)
	#pdb.set_trace()
	for i in range(len(patches)):
		patches[i] = (patches[i] - np.mean(patches[i]))/np.std(patches[i])

	return patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], 1)

def gen_keypoints_from_csv(csv):
	keypoints = []
	for line in csv:
		if line['valid'] == 1:
			k = cv2.KeyPoint(line['x'], line['y'],7.0, 0.0)
			k.class_id = int(line['id'])
			keypoints.append(k)

	return keypoints	 

			
def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]
	return dirs or False

def save_dist_matrix(ref_kps, ref_descriptors, ref_gt, tgt_kps, descriptors, tgt_gt, out_fname):
	#np.linalg.norm(a-b)
	print ('saving matrix in:',  out_fname)
	size = len(ref_gt)
	dist_mat = np.full((size,size),-1.0,dtype = np.float32)
	valid_m = 0
	matches=0

	matching_sum = 0

	begin = time.time()

	for m in range(len(ref_kps)):
		i = ref_kps[m].class_id
		if ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			valid_m+=1
		for n in range(len(tgt_kps)):
			j = tgt_kps[n].class_id
			if ref_gt[i]['valid'] and tgt_gt[i]['valid'] and tgt_gt[j]['valid']:
				dist_mat[i,j] = np.linalg.norm(ref_descriptors[m]-descriptors[n]) #distance.euclidean(ref_d,tgt_d) #np.linalg.norm(ref_d-tgt_d)

	print('Time to match geopatch: %.3f'%(time.time() - begin))

	mins = np.argmin(np.where(dist_mat >= 0, dist_mat, 65000), axis=1)
	for i,j in enumerate(mins):
		if i==j and ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			matches+=1

	print ('--- CORRECT MATCHES --- %d/%d'%(matches,valid_m))

	with open(out_fname, 'w') as f:

		f.write('%d %d\n'%(size,size))

		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.8f '%(dist_mat[i,j]))



def extract(args):

	ref_descriptor = None
	ref_gt = None


	if args.file:
		exp_list = get_dir_list(args.input)
	elif args.dir:
		exp_list = [d for d in glob.glob(args.input+'/*') if os.path.isdir(d)]
	else:
		exp_list = [args.input]

	for exp_dir in exp_list:

		dataset_name = os.path.abspath(exp_dir).split('/')[-1]

		experiment_files = glob.glob(exp_dir + "/*RAW*")

		#print(experiment_files) ; input()
	
		master_f = ''
		for exp_file in experiment_files:
			if 'master' in exp_file or 'ref' in exp_file:
				fname = exp_file.split('_RAW_')[0]
				#print(fname) ; input()
				ref_gt = np.recfromcsv(fname + '.csv', delimiter =',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
				correct_cadar_csv(ref_gt)
				ref_kps = gen_keypoints_from_csv(ref_gt)
				ref_patches = toTFShape(load_raw_patches(exp_file))
				if len(ref_kps) != len(ref_patches):
					print("Error: Nb. of patches does not match the nb. of keypoints!"); exit(0)

				ref_descriptors = net.model.predict(ref_patches, batch_size=300)
				master_f = exp_file

		for exp_file in experiment_files:

			if 'master' not in exp_file and 'ref' not in exp_file:
				fname = exp_file.split('_RAW_')[0]
				tgt_gt = np.recfromcsv(fname + '.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
				correct_cadar_csv(tgt_gt)
				tgt_kps = gen_keypoints_from_csv(tgt_gt)
				tgt_patches = toTFShape(load_raw_patches(exp_file))
				if len(tgt_kps) != len(tgt_patches):
					print("Error: Nb. of patches does not match the nb. of keypoints!"); exit(0)

				begin = time.time()
				descriptors = net.model.predict(tgt_patches, batch_size=300)
				print('Time to extract geopatch: %.3f'%(time.time() - begin))
				mat_fname = os.path.basename(master_f).split('_RAW_')[0] + '__' + os.path.basename(exp_file).split('_RAW_')[0] + \
							'__' + 'GEOPATCHCNN.txt'

				result_dir = args.input + '/' + os.path.basename(exp_dir)#os.path.join(args.output,experiment_name) + '/' + dataset_name + '/' + exp_dir_target
				check_dir(result_dir)
				#print(result_dir) ; input()
				#ref_descriptors, ref_gt = descriptors, tgt_gt
				save_dist_matrix(ref_kps,ref_descriptors,ref_gt, tgt_kps, descriptors,tgt_gt, os.path.join(result_dir,mat_fname))


extract(parseArg())
