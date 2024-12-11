import os
import subprocess
import glob
import argparse
import numpy as np
import numpy.random as rng
import re
import multiprocessing
import time
import tqdm

#rng.seed(57)
rng.seed(200)

sim_bin_folder = '/homeLocal/guipotje/sim_build/'

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]

	return dirs

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input path containing images"
	, required=False, default = './') 
	parser.add_argument("-o", "--output", help="Output path where results will be saved."
	, required=False, default = '/homeLocal/guipotje/sim_test') 

	args = parser.parse_args()
	return args


def run_sim_parallel(command):
	proc = subprocess.Popen(['/bin/bash','-c',command])	
	proc.wait()


def run_sim(args):

	images = glob.glob(args.input + "/*.png") + glob.glob(args.input + "/*.jpg") + glob.glob(args.input + "/*.bmp") 

	commands = []
	
	for i, img_path in enumerate(images):

		img_name = os.path.splitext(os.path.basename(os.path.abspath(img_path)))[0]
	
		fx_var, fy_var, fz_var = 0 , 0, 0
		gravity = 0.4 #1.

		# Wind Force
		if rng.random() < 0.7: fx_var = rng.uniform(0.3,0.4)#(0.5, 2.3)
		if rng.random() < 0.7: fy_var = rng.uniform(0.3,0.4)#(0.5, 2.3)
		if rng.random() < 0.7: fz_var = rng.uniform(0.1,0.1)#(0.5, 2.3)

		var_int = rng.randint(10,20)

		out_dir = os.path.abspath(args.output) + '/' + 'DATASET_' + img_name 

		#building command
		command = sim_bin_folder + './nonrigid_sim_auto '

		command+= img_path + ' '
		command+= '--out_dir ' + out_dir + ' '
		command+= '--save '

		### Disable Deformation if uncommented ###
		#fx_var, fy_var, fz_var, gravity = 0,0,0,0
		#command+= '--fz ' + "{:.5f} ".format(0)
		##########################################

		command+= '--fx_var ' + "{:.5f} ".format(fx_var)
		command+= '--fy_var ' + "{:.5f} ".format(fy_var)
		command+= '--fz_var ' + "{:.5f} ".format(fz_var)

		command+= '--gravity ' + "{:.5f} ".format(gravity)

		command+= '--variation_interval ' + "{} ".format(var_int)

		command+= '--light_var 0 ' #1
		command+= '--rot_var 0 ' #15 in degrees

		command+= '--cx 0 ' #1.6
		command+= '--cy 0 ' #1.6

		command+= '--seed ' + "{} ".format(i+250)

		command+= '--save_interval 3 ' #100
		command+= '--save_range 0 ' #60
		command+= '--wait_time ' + "{:d} ".format(200)#rng.randint(500))

		command+= '--max_frames 30 '

		command+= '--width 960 '
		command+= '--height 720 '

		command+= '--background ' + images[rng.randint(len(images))] + ' '

		command+= '--verbose true '

		#print(command) ; input()
		commands.append(command)

		check_dir(out_dir)
		#print command; raw_input()


	#print(commands); input()
	p = multiprocessing.Pool(processes=4)
	#p.map(run_sim_parallel,commands)

	for _ in tqdm.tqdm(p.imap_unordered(run_sim_parallel, commands), total=len(commands)):
	    pass

args = parseArg()
run_sim(args)
