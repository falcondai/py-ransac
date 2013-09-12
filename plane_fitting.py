import numpy as np
from matplotlib import pylab

def augment(xyzs):
	axyz = ones((len(xyzs), 4))
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
	n = 100
	max_iteration = 100
	goal_inliers = n * 0.3

	# test data
	xyzs = np.random.random((n, 3)) * 10
	xyzs[:50, 2:] = xyzs[:50, :1]

	# pylab.scatter(xyzs.T[0], xyzs.T[1])

	# RANSAC
	best = 0
	best_model = None
	for i in xrange(max_iteration):
		s = sample(xyzs, 3)
		m = estimate(s)
		ic = count_inliers(xyzs, m, 0.01)
		
		print s
		print 'estimate:', m,
		print '# inliers:', ic
		a, b, c, d = m
		# pylab.plot([0, 10], [-c/b, -(c+10*a)/b])
		
		if ic > best:
			best = ic
			best_model = m
			# pylab.scatter(s.T[0], s.T[1], color='red')
			if ic > goal_inliers:
				break
	print 'took iterations:', i+1, 'best model:', best_model, 'explains:', best