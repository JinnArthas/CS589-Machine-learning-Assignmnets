# Import modules
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import random
import pprint

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
	# Function to perform k_mean_clustering and calculating the 
	# SSE for each pixel with its centroid 
	# input is image and k 
	# returns flatter reconstructed image, reconstructed image
	# and array of sse of pixels to its centroid.
	clf = KMeans(n_clusters=k)
	clf.fit(flattened_image)
	labels = clf.predict(flattened_image)
	print("Fitting and predicting done for k " + str(k))
	re_image = np.zeros((flattened_image.shape))
	squared_error = 0
	for pixel in range(flattened_image.shape[0]):
		cluster_center = clf.cluster_centers_[labels[pixel]]
		squared_error += mean_squared_error(flattened_image[pixel], cluster_center)
		re_image[pixel] = cluster_center
	re_image = re_image/255
	return re_image, re_image.reshape(400, 400, 3), np.log(squared_error)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
	K = [2 ,5,10 ,25,50,75 ,100,200]
	"""
	result = []
	SSE = []
	f, ax = plt.subplots(3, 3)
	ax[0, 0].imshow(data_x)
	ax[0, 0].set_title("Original Image", fontsize=6)
	for i, k in enumerate(K):
		i = i + 1
		re_image, new_image, squared_error = k_mean_clustering(k, flattened_image)
		result.append(re_image); SSE.append(squared_error)
		#print(int((i - i%3)/3), int(i - int(i/3)*3))
		ax[int((i - i%3)/3), int(i - int(i/3)*3)].imshow(new_image)
		ax[int((i - i%3)/3), int(i - int(i/3)*3)].set_title("Reconstructed image for k - " + str(k), fontsize = 6)
	plt.tight_layout()
	"""
	"""
	#plt.savefig("../Figures/Reconstructed_image.png")
	#pickle.dump(result, open( "save" + str(random.random) + ".p", "wb" ) )
	"""
	recons_image = pickle.load( open( "save.p", "rb" ) )
	recons_error = []
	for i, k in enumerate(K):
		err = rmse(recons_image[i], flattened_image)
		recons_error.append((k,err))

	print("*"*10 + "Reconstruction Error" + "*"*10)
	pprint.pprint(recons_error)


	print("*"*10 + "Compresseion Ratio"+ "*"*10)
	for k in K:
		pixels = data_x.shape[0] * data_x.shape[1]
		tmp = (24*k + pixels*np.log2(k))/(24*pixels)  
		print("For K equals: " + str(k) + " compression ratio is: "  + str(tmp))
	
	# Just having it here so that I dont have to open 
	SSE = [19.364452852413656, 18.284265383653125, 17.592396941810147, 16.774010174561404, 16.223725553550821, 15.9212126963522, 15.706185410704739, 15.233038289366752]
	print("*"*10 + "Elbow Graph" + "*"*10)
	plt.scatter(K, SSE, c='r')
	plt.plot(K, SSE)
	plt.ylabel("Sum of Squared error (log)")
	plt.xlabel("Number of cluster (k)")
	plt.title("Sum of Squared Error of Each Pixel from Centroids")
	plt.savefig("../Figures/kmeanElbow.png")
	plt.show()

	reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	print('Reconstructed image = ', reconstructed_image.shape)