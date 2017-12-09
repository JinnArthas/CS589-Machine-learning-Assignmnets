# Import modules
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import random

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

def k_mean_clustering(k, flattened_image):
	clf = KMeans(n_clusters=k)
	clf.fit(flattened_image)
	labels = clf.predict(flattened_image)
	print("Fitting and predicting done")
	re_image = np.copy(flattened_image)
	for pixel in range(flattened_image.shape[0]):
		tmp = clf.cluster_centers_[labels[pixel]]
		re_image[pixel] = clf.cluster_centers_[labels[pixel]]
	return re_image, re_image.reshape(400, 400, 3)


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
	#plt.imshow(data_x)
	#plt.show()
	flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	print('Flattened image = ', flattened_image.shape)
	#flattened_image= np.array(flattened_image, dtype=np.float64) / 255
	#print(flattened_image)
	print('Implement k-means here ...')

	K = [2,10,25,50,75 ,100,200]
	result = []
	f, ax = plt.subplots(3, 3)
	ax[0, 0].imshow(data_x)
	ax[0, 0].set_title("Original Image", fontsize=6)
	for i, k in enumerate(K):
		i = i + 1
		re_image, new_image = k_mean_clustering(k, flattened_image)
		result.append(re_image)
		#print(int((i - i%3)/3), int(i - int(i/3)*3))
		ax[int((i - i%3)/3), int(i - int(i/3)*3)].imshow(new_image)
		ax[int((i - i%3)/3), int(i - int(i/3)*3)].set_title("Reconstructed image for k equals " + str(k), fontsize = 6)
	plt.tight_layout()
	plt.savefig("../Figures/Reconstructed_image_k_" + str(k) + ".png")
	pickle.dump(result, open( "save" + str(random.random) + ".p", "wb" ) )
	reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	print('Reconstructed image = ', reconstructed_image.shape)

