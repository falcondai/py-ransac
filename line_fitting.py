import numpy as np

def augment(xys):
	axy = np.ones((len(xys), 3))
	axy[:, :2] = xys
	return axy

def sample(xys, n):
	np.random.shuffle(xys)
	return xys[:n]

def estimate(xys):
	axy = augment(xys)
	return np.linalg.svd(axy)[-1][-1, :]

def count_inliers(xys, coeffs, threshold):
	return len(filter(lambda e: np.abs(e) < threshold, coeffs.dot(augment(xys).T)))



if __name__ == '__main__':
	from matplotlib import pylab

	n = 100
	max_iterations = 100
	goal_inliers = n * 0.3

	# test data
	xys = np.random.random((n, 2)) * 10
	xys[:50, 1:] = xys[:50, :1]

	pylab.scatter(xys.T[0], xys.T[1])

	# RANSAC
	best = 0
	best_model = None
	for i in xrange(max_iterations):
		s = sample(xys, 2)
		m = estimate(s)
		ic = count_inliers(xys, m, 0.01)
		
		print s
		print 'estimate:', m,
		print '# inliers:', ic
		a, b, c = m
		pylab.plot([0, 10], [-c/b, -(c+10*a)/b])
		
		if ic > best:
			best = ic
			best_model = m
			# pylab.scatter(s.T[0], s.T[1], color='red')
			if ic > goal_inliers:
				break
	print 'took iterations:', i+1, 'best model:', best_model, 'explains:', best