import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, glob, argparse, tqdm
from scipy.spatial import cKDTree as KDTree
import numpy.random as rng
import pdb
import multiprocessing
import h5py

np.set_printoptions(suppress=True)
rng.seed(57)

detector = cv2.SIFT_create(nfeatures = 3000, contrastThreshold=0.02)
args = None

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input path containing images"
	, required=True) 
	parser.add_argument("-o", "--output", help="Output path where results will be saved."
	, required=True) 
	parser.add_argument("-n", "--npairs", help="Number of pairs to sample.", type=int
	, required=False, default = 10) 
	parser.add_argument("--createh5", help="Create H5 dataset instead of building raw dataset"
	, action = 'store_true') 

	args = parser.parse_args()
	return args

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def write_sift(filepath, kps):
	with open(filepath + '.sift', 'w') as f:
		f.write('size, angle, x, y, octave\n')
		for kp in kps:
			f.write('%.2f, %.3f, %.2f, %.2f, %d\n'%(kp.size, kp.angle, kp.pt[0], kp.pt[1], kp.octave))

def stack_geopatches(geopatches):
	psize = 32
	stacked = geopatches[:, :32, np.newaxis]

	for i in range(1, geopatches.shape[1] // psize):
		p = geopatches[:, i*psize : (i+1)*psize, np.newaxis]
		stacked = np.concatenate((stacked, p), axis=2)

	if stacked.shape[-1] != geopatches.shape[1] // psize:
		raise RuntimeError('Sizes does not match!')

	return stacked

def create_h5dataset(raw_dataset_path, out_path):

	dataset = h5py.File(out_path, "w")
	dataset.create_group('imgs')
	dataset.create_group('kps')
	dataset.create_group('geopatch')

	for pair in tqdm.tqdm(glob.glob(raw_dataset_path + '/*-rgb.png')):
		pair = pair.replace('-rgb','')
		pair_id, ext = os.path.splitext(pair)
		#print(pair_id) ; input() ; quit()
		img = cv2.imread(pair_id +'-rgb.png')
		theta = np.load(pair_id + '.npy')
		geopatch = stack_geopatches( cv2.imread(pair_id +'-geopatch.png', 0) )
		pair_id = os.path.basename(pair_id)
		dataset['imgs'].create_dataset(pair_id, data = img, compression="gzip", compression_opts=9)
		dataset['kps'].create_dataset(pair_id, data = theta, compression="gzip", compression_opts=9)
		dataset['geopatch'].create_dataset(pair_id, data = geopatch, compression="gzip", compression_opts=9)

def random_pairs(npairs, N):
	M = np.dstack(np.meshgrid(np.arange(N), np.arange(N)))
	mask = np.tril(np.ones_like(M[...,0]), k=-1)
	#print(mask)
	pairs = M[mask==1]
	np.random.shuffle(pairs)
	#print(pairs)
	return pairs[:npairs]

#print(random_pairs(20,20))

def reproject_points(grid, K, H):
	#reproject grid points using intrinsics
	grid = np.copy(grid)
	grid[:,:,0]/= grid[:,:,2] ; grid[:,:,1]/= grid[:,:,2]
	grid[:,:,0]*=K[0,0] ; grid[:,:,1]*=K[1,1]
	grid[:,:,0]+=K[0,2] ; grid[:,:,1]+=K[1,2]
	grid[:,:,1] = H-1-grid[:,:,1]
	#pts = grid[grid[:,:,3] == 1.0]
	#pts = grid[:,:,:2].reshape(-1,2)
	return grid

def gen_trimesh_idx(H,W):
	# @Brief:
	# Generates a triangle mesh configured as shown below:
	# (x,y)   *--* (x+1,y)
	#         | /|
	#         |/ |
	# (x,y+1) *--* (x+1,y+1)

	X, Y = np.meshgrid(np.arange(W-1), np.arange(H-1))
	X1 = X + 1
	Y1 = Y + 1

	UpperT = np.dstack((X,Y,X1,Y,X,Y1))
	LowerT = np.dstack((X1,Y1,X1,Y,X,Y1))

	UpperT = UpperT.reshape(-1,6).reshape(-1,3,2)
	LowerT = LowerT.reshape(-1,6).reshape(-1,3,2)
	return np.vstack((UpperT,LowerT)) # stack upper and lower triangles in a single array


def gen_bounding_squares(mesh2d):
	# -----------
	# Params:
	# mesh2d: 2d mesh array of shape (N,3,2)
	mins = np.min(mesh2d, axis=1)
	maxs = np.max(mesh2d, axis=1)

	mids = (mins + maxs) / 2

	return mids, np.max((maxs - mins)/2, axis=1)

def find_barycentric_coords(grid, mesh_idx, K, img):
	'''
	@Brief:
	Given a grid of 3D points, reproject them in the image, triangulate the grid. Then, find SIFT keypoints,
	and calculate barycentric coords
	-----
	Params:
	grid: (H,W,4) array of 3D points and flag if they are not occluded in last channel (1), (0) otherwise.
	mesh_idx: array containing the faces indices of shape (N,3,2) where second dim is the three triangle vertices and last dim
				the (x,y) idx coordinates in the original grid
	K: (3,3) Camera matrix
	img: RGB image
	-----
	Returns:
	face: Dict containing for each face the index of face in the grid as a key and keypoints with their barycentric coords
		  keypoints: list of cv2.KeyPoint
	''' 

	img_grid = reproject_points(grid, K, img.shape[0])
	pts = img_grid.reshape(-1,4)

	#print(mesh_idx.shape) ; input()
	mesh_idx_flat = mesh_idx.reshape(-1,2)
	xx = mesh_idx_flat[:,0]
	yy = mesh_idx_flat[:,1]
	mesh2d = img_grid[yy, xx]
	mesh2d_xy = mesh2d[..., :2]
	mesh2d = mesh2d.reshape(-1,3,4)

	# for i in range(0, len(mesh_img), 3):
	# 	x1,y1 = mesh_img[i]
	# 	x2,y2 = mesh_img[i+1]
	# 	x3,y3 = mesh_img[i+2]
	# 	plt.plot([x1,x2,x3,x1], [y1,y2,y3,y1], c='blue')

	mesh2d_xy = mesh2d_xy.reshape(-1,3,2)
	centers, radii = gen_bounding_squares(mesh2d_xy)

	# plt.scatter(pts[:,0]s, pts[:,1], c='red', s=1)
	# rdidx = rng.randint(0,centers.shape[0],500) ;
	# for i in rdidx:
	# 	cv2.rectangle(img, (centers[i,0]-radii[i], centers[i,1]-radii[i]), 
	# 						(centers[i,0]+radii[i], centers[i,1]+radii[i]), (0,0,255), 1)
	# 	cv2.line(img, tuple(mesh2d[i,0]), tuple(mesh2d[i,1]), (255, 0, 0), 1)
	# 	cv2.line(img, tuple(mesh2d[i,1]), tuple(mesh2d[i,2]), (255, 0, 0), 1)
	# 	cv2.line(img, tuple(mesh2d[i,2]), tuple(mesh2d[i,0]), (255, 0, 0), 1)

	kps = detector.detect(img, None)
	kps_pt = [kp.pt for kp in kps]
	tree = KDTree(kps_pt)

	query = tree.query_ball_point(centers, 1.42*radii, p=2)
	faces = {}

	for i, q in enumerate(query):
		if np.all(mesh2d[i,:,3] == 1): #if triangle is not occluded
			for idx in q:
				pt = kps_pt[idx]
				#calculate baricentric coords
				T = np.array([ [mesh2d_xy[i,0,0] - mesh2d_xy[i,2,0], mesh2d_xy[i,1,0] - mesh2d_xy[i,2,0]],
							   [mesh2d_xy[i,0,1] - mesh2d_xy[i,2,1], mesh2d_xy[i,1,1] - mesh2d_xy[i,2,1]] ])
				Ls = np.linalg.inv(T)@(pt - mesh2d_xy[i,2])
				# print(pt)
				# print(Ls[0]* mesh2d_xy[i,0] +  Ls[1]* mesh2d_xy[i,1] + (1.0 - Ls[0] - Ls[1]) * mesh2d_xy[i,2])
				# input()

				if np.all(Ls>=0) and np.all(Ls <=1.0) and np.sum(Ls) <=1: # Check if the point is inside the triangle
					if i not in faces:
						faces[i] = [{'idx': idx, 'lambda':Ls}]
					else:
						faces[i].append({'idx': idx, 'lambda':Ls})
				#print(Ls) ; input()

	#plot result
	# for p in kps_pt:
	# 	cv2.circle(img, (int(p[0]), int(p[1])), 4, (0,0,255), 2)

	# for k, v in faces.items():
	# 	for kps in v:
	# 		idx = kps['idx']
	# 		cv2.circle(img, (int(kps_pt[idx][0]), int(kps_pt[idx][1])), 4, (0,255,0), 2)

	# plt.figure(figsize = (10,10))
	# plt.imshow(img[...,::-1])
	# plt.show()
	# quit()

	return faces, kps

def match_pair(pairs):

	img1_path, img2_path, pair_id = pairs

	img1 = cv2.imread(img1_path)
	img2 = cv2.imread(img2_path)
	fsK = cv2.FileStorage(os.path.dirname(img1_path) + "/intrinsics.xml", cv2.FILE_STORAGE_READ)
	fsG1 = cv2.FileStorage(img1_path.split('-rgb')[0] + "-grid.xml", cv2.FILE_STORAGE_READ)
	fsG2 = cv2.FileStorage(img2_path.split('-rgb')[0] + "-grid.xml", cv2.FILE_STORAGE_READ)
	depth1 = cv2.imread(img1_path.split('-rgb')[0] + "-depth.png", cv2.IMREAD_ANYDEPTH )
	depth2 = cv2.imread(img2_path.split('-rgb')[0] + "-depth.png", cv2.IMREAD_ANYDEPTH )

	if not fsK.isOpened() or not fsG1.isOpened() or not fsG2.isOpened() or img1.shape[0]==0 or img2.shape[0]==0:
		raise RuntimeError('Not able to open required files!')

	K = fsK.getNode("intrinsics").mat()
	grid1 = fsG1.getNode('grid').mat()
	grid2 = fsG2.getNode('grid').mat()

	mesh_idx = gen_trimesh_idx(*grid1.shape[:2])

	faces1, kps1 = find_barycentric_coords(grid1, mesh_idx, K, img1)
	kps2 = detector.detect(img2, None)

	tree = KDTree([k.pt for k in kps2]) #Initialize a KDTree for the second img keypoints
	pts = []
	pts_idx = []
	for face_id, keypoints in faces1.items():
		for kp in keypoints: #for each kp, use barycentric coords to reproject the keypoint to the target frame
			A = grid2[tuple(mesh_idx[face_id, 0][::-1])][:3]
			B = grid2[tuple(mesh_idx[face_id, 1][::-1])][:3]
			C = grid2[tuple(mesh_idx[face_id, 2][::-1])][:3]
			l1, l2 = kp['lambda']
			X = l1 * A +  l2 * B +  (1.0 - l1 - l2)* C
			pts.append(X)
			pts_idx.append(kp['idx'])

	#reproject
	pts = np.array(pts); pts_idx = np.array(pts_idx)

	if len(pts.shape) == 2:
		pts[:,0]/= pts[:,2]
		pts[:,1]/= pts[:,2]
		pts[:,2] = 1.0
		pts = (K @ pts.T).T
		pts[:,1] = img2.shape[0] - 1 - pts[:,1]

		#DRAW to test if reprojection is correct
		# rdidx = rng.randint(0,len(pts_idx),30)
		# plot_kps1 = [kps1[i] for i in pts_idx[rdidx]]
		# plot_kps2 = [cv2.KeyPoint(p[0], p[1], 7.) for p in pts[rdidx]]
		# dmatch = [cv2.DMatch(i,i,1) for i in range(len(plot_kps1))]
		# img_match = cv2.drawMatches(img1, plot_kps1, img2, plot_kps2, dmatch,
		# 	                        None, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# plt.figure(figsize = (10,10))
		# plt.imshow(img_match[...,::-1])
		# plt.show()
		# quit()
		
		dists, qidx = tree.query(pts[:,:2])
		threshold = 3.0
		kps1 = [kps1[i] for i in pts_idx] #filter unused keypoints
		inliers1 = np.arange(len(kps1))[dists < threshold] 
		inliers2 = qidx[dists < threshold]

		kps1 = [kps1[i] for i in inliers1]
		kps2 = [kps2[i] for i in inliers2]

		#print('-----> ', len(kps1))

		#Draw match for verification
		# dmatch = [cv2.DMatch(i,i,1) for i in range(len(kps1))]
		# img_match = cv2.drawMatches(img1, kps1, img2, kps2, dmatch,
		# 	                        None, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imwrite('/homeLocal/guipotje/hd_sim/out.png', img_match)
		# plt.figure(figsize = (10,10))
		# plt.imshow(img_match[...,::-1])
		# plt.show()	
		# quit()

		if len(kps1) >= 50:
			#filter too large kps
			max_s = 5.
			f_kps1 = kps1 #[kps1[i] for i in range(len(kps1)) if kps1[i].size < max_s and kps2[i].size < max_s]
			f_kps2 = kps2 #[kps2[i] for i in range(len(kps2)) if kps1[i].size < max_s and kps2[i].size < max_s]

			theta1 = np.array([(k.pt[0], k.pt[1], k.angle, k.size) for k in f_kps1], dtype = np.float32)
			theta2 = np.array([(k.pt[0], k.pt[1], k.angle, k.size) for k in f_kps2], dtype = np.float32)

			cv2.imwrite(args.output + '/' + pair_id + '__1-rgb.png', img1)
			cv2.imwrite(args.output + '/' + pair_id + '__2-rgb.png', img2)
			cv2.imwrite(args.output + '/' + pair_id + '__1-depth.png', depth1)
			cv2.imwrite(args.output + '/' + pair_id + '__2-depth.png', depth2)
			np.save(args.output + '/' + pair_id + '__1.npy', theta1)
			np.save(args.output + '/' + pair_id + '__2.npy', theta2)
			write_sift(args.output + '/' + pair_id + '__1', f_kps1)
			write_sift(args.output + '/' + pair_id + '__2', f_kps2)

	else:
		print("Warning! Array bad shape: ", pts.shape)
		print("From: ")
		print("   Pair1: ", img1_path)
		print("   Pair2: ", img2_path)


def main():
	global args
	args = parseArg()

	if args.createh5:
		create_h5dataset(raw_dataset_path = args.input, out_path = args.output)

	else:
		datasets = glob.glob(args.input + "/DATASET*")
		pool = multiprocessing.Pool(processes=30)

		pairs = []

		for d in datasets:
			imgs = glob.glob(d + "/*-rgb.png")
			pairs_idx = random_pairs(args.npairs, len(imgs))
			pairs_d = [ (imgs[p[0]], imgs[p[1]], '{:s}__{:02d}'.format(d.split('DATASET_')[1],i)) \
															for i, p in enumerate(pairs_idx) ]
			pairs+=pairs_d

		if len(pairs) > 0:
			check_dir(args.output)

		#match_pair(pairs[50])

		for _ in tqdm.tqdm(pool.imap_unordered(match_pair, pairs), total=len(pairs)):
			pass

main()

