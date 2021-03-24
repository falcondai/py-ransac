
import numpy as np

def run_ransac(data,
                 estimate,
                 inlier_threshold,
                 sample_size=3,
                 goal_inliers=0.3,
                 max_iterations=100,
                 stop_at_goal=True,
                 random_seed=None):
    best_ic = 0
    best_model = None
    for i in range(max_iterations):
        s = data[np.random.choice(data.shape[0], int(sample_size)), :]
        m = estimate(s)
        ic = count_inliers(m, data, inlier_threshold)

        print(s)
        print('estimate:', m,)
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic


def plane_points_distances(plane, points):
    aug_points = np.ones((points.shape[0], 4))
    aug_points[:, :-1] = points
    return np.abs(np.dot(plane, aug_points.T))


def count_inliers(plane, points, inlier_threshold):
    return np.count_nonzero(plane_points_distances(plane, points) < inlier_threshold)

