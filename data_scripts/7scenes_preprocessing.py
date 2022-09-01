# Code from https://github.com/tsattler/visloc_pseudo_gt_limitations/

import os
import warnings

import numpy as np
from skimage import io
from joblib import Parallel, delayed

# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '/mnt/res_nas/shared/datasets/7scenes'
focal_length = 525.0

# focal length of the depth sensor (source: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
d_focal_length = 585.0

# RGB image dimensions
img_w = 640
img_h = 480

# sub sampling factor of eye coordinate tensor
nn_subsampling = 8

#transformation from depth sensor to RGB sensor
#calibration according to https://projet.liris.cnrs.fr/voir/activities-dataset/kinect-calibration.html
d_to_rgb = np.array([
	[ 9.9996518012567637e-01,  2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
	[-2.7409311281316700e-03,  9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
	[ 7.8819942130445332e-03,  8.1718328771890631e-03,  9.9993554558014031e-01, 2.0318321729487039e-03],
	[0, 0, 0, 1]
])

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

# download the original 7 scenes dataset for poses and images
# mkdir(src_folder)
# os.chdir(src_folder)

# for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:

# 	print("=== Downloading 7scenes Data:", ds, "===============================")

# 	os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
# 	os.system('unzip ' + ds + '.zip')
# 	os.system('rm ' + ds + '.zip')

# 	sequences = os.listdir(ds)

# 	for file in sequences:
# 		if file.endswith('.zip'):

# 			print("Unpacking", file)
# 			os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
# 			os.system('rm ' + ds + '/' + file)

# print("Processing frames...")

def process_scene(ds):

	def process_frames(split_file):

		# read the split file
		with open(ds + '/' + split_file, 'r') as f:
			split = f.readlines()
		# map sequences to folder names
		split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

		for seq in split:
			files = os.listdir(ds + '/' + seq)

			# adjust depth files by mapping to RGB sensor
			depth_files = [f for f in files if f.endswith('depth.png')]

			for d_index, d_file in enumerate(depth_files):
				if d_index % 1000 == 0: 
					print(d_index, ds, split_file)

				depth = io.imread(ds + '/' + seq + '/' + d_file)
				depth = depth.astype(np.float32)
				depth /= 1000  # from millimeters to meters

				d_h = depth.shape[0]
				d_w = depth.shape[1]

				# reproject depth map to 3D eye coordinates
				eye_coords = np.zeros((4, d_h, d_w))
				# set x and y coordinates
				eye_coords[0] = 0.5 + np.dstack([np.arange(0, d_w)] * d_h)[0].T
				eye_coords[1] = 0.5 + np.dstack([np.arange(0, d_h)] * d_w)[0]

				eye_coords = eye_coords.reshape(4, -1)
				depth = depth.reshape(-1)

				# filter pixels with invalid depth
				depth_mask = (depth > 0) & (depth < 100)
				eye_coords = eye_coords[:, depth_mask]
				depth = depth[depth_mask]

				# substract depth principal point (assume image center)
				eye_coords[0] -= d_w / 2
				eye_coords[1] -= d_h / 2
				# reproject
				eye_coords[0:2] /= d_focal_length
				eye_coords[0] *= depth
				eye_coords[1] *= depth
				eye_coords[2] = depth
				eye_coords[3] = 1

				# transform from depth sensor to RGB sensor
				eye_coords = np.matmul(d_to_rgb, eye_coords)

				# project
				depth = eye_coords[2]

				eye_coords[0] /= depth
				eye_coords[1] /= depth
				eye_coords[0:2] *= focal_length

				# add RGB principal point (assume image center)
				eye_coords[0] += img_w / 2
				eye_coords[1] += img_h / 2

				registered_depth = np.ones((img_h, img_w), dtype=np.float32) * 2e3

				for pt in range(eye_coords.shape[1]):
					x = round(eye_coords[0, pt])
					y = round(eye_coords[1, pt])
					d = eye_coords[2, pt]

					if x < 0 or y < 0 or y >= d_h or x >= d_w:
						continue

					registered_depth[y, x] = min(registered_depth[y, x], d)

				registered_depth[registered_depth > 1e3] = 0
				registered_depth = (1000 * registered_depth).astype(np.uint16)

				# store calibrated depth
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					io.imsave(ds + '/' + seq + '/' + d_file.replace("depth.png", "depth.proj.png"), registered_depth)

	process_frames('TrainSplit.txt')
	process_frames('TestSplit.txt')

Parallel(n_jobs=7, verbose=0)(
	map(delayed(process_scene), [os.path.join(src_folder, scan_name) for scan_name in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']]))