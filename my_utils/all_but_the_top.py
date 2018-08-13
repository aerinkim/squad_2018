import numpy as np
from sklearn.decomposition import PCA
import scipy

def creat_np_array_from_Glove():
	arr = np.zeros(shape=(2196017,300))
	with open('../data/glove.840B.300d.txt', encoding="utf8") as f:
		i=0
		token_list = []
		for line in f:
			elems = line.split()
			arr[i] = [float(v) for v in elems[-300:]]
			i+=1
			token = ' '.join(elems[0:-300])
			token_list.append(token)
	return arr, token_list


def all_but_the_top(v):
	mu = np.mean(v, axis=0) # average of all vectors. collapse on axis = 0. 
	v_tilde = v - mu  
	D=2
	
	#pca = PCA(n_components=D)
	#u = pca.fit_transform(v.T)

	X = v_tilde
	cov_matrix = X.T.dot(X)/X.shape[0]
	# Run single value decomposition to get the U principal component matrix
	U, S, V = scipy.linalg.svd(cov_matrix, full_matrices = True, compute_uv = True)
	u = U

	# Postprocess
	for w in range(v_tilde.shape[0]):
		for i in range(D):
			v_tilde[w, :] = v_tilde[w, :] - u[:, i].dot(v[w]) * u[:, i].T

	print ("sanity check (v_tilde[:10]) :", v_tilde[:10])
	return v_tilde


def write_a_post_processed_GloVe(arr, token_list):
	with open("../data/all_but_the_top.840B.txt", mode="w" , encoding="utf8") as outfile: 
		for i in range(0,len(token_list)):
			token = token_list[i]
			dim_300 = ' '.join(str(j) for j in arr[i]) 
			outfile.write("%s %s\n" %(token, dim_300))
			if i % 100000 == 0:
				print ("write file:%s %s %s\n" %(str(i), token, dim_300))


if __name__ == '__main__':
	arr, token_list = creat_np_array_from_Glove()
	v_tilde = all_but_the_top(arr)
	write_a_post_processed_GloVe(v_tilde, token_list)
