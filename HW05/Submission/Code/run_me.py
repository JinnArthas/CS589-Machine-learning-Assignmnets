# Import modules

import numpy as np
from scipy import misc

def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')

	return (data_x)

def read_faces():
	nFaces = 100
	nDims = 2500
	data_x = np.empty((0, nDims), dtype=float)

	for i in np.arange(nFaces):
		data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))

	return (data_x)

if __name__ == '__main__':
	
	################################################
	# PCA

	data_x = read_faces()
	print('X = ', data_x.shape)

	print('Implement PCA here ...')
	
	################################################
	# K-Means

	data_x = read_scene()
	print('X = ', data_x.shape)

	flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	print('Flattened image = ', flattened_image.shape)

	print('Implement k-means here ...')

	reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	print('Reconstructed image = ', reconstructed_image.shape)

