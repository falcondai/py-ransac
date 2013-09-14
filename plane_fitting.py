import numpy as np

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def sample(xyzs, n):
	np.random.shuffle(xyzs)
	return xyzs[:n]

def estimate(xyzs):
	axyz = augment(xyzs)
	return np.linalg.svd(axyz)[-1][-1, :]

def count_inliers(xyzs, coeffs, threshold):
	return len(filter(lambda e: np.abs(e) < threshold, coeffs.dot(augment(xyzs).T)))



if __name__ == '__main__':
	from matplotlib import pylab
	from mpl_toolkits import mplot3d
	fig = pylab.figure()
	ax = mplot3d.Axes3D(fig)

	def plot_plane(a, b, c, d):
		xx, yy = np.mgrid[4:6, 4:6]
		return xx, yy, (-d - a * xx - b * yy) / c

	n = 100
	max_iterations = 100
	goal_inliers = n * 0.3

	# test data
	xyzs = np.random.random((n, 3)) * 10
	xyzs[:50, 2:] = xyzs[:50, :1]

	ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

	# RANSAC
	best = 0
	best_model = None
	for i in xrange(max_iterations):
		s = sample(xyzs, 3)
		m = estimate(s)
		ic = count_inliers(xyzs, m, 0.01)
		
		print s
		print 'estimate:', m,
		print '# inliers:', ic
		a, b, c, d = m
		xx, yy, zz = plot_plane(a, b, c, d)
		ax.plot_surface(xx, yy, zz)
		
		if ic > best:
			best = ic
			best_model = m
			# ax.scatter3D(s.T[0], s.T[1], s.T[2], color='red')
			if ic > goal_inliers:
				break
	print 'took iterations:', i+1, 'best model:', best_model, 'explains:', best