import numpy as np
from sklearn.cluster import KMeans

######################### K-Medoids

def assign_nearest(ids_of_mediods):
    dists = dist(x[:,None,:], x[None,ids_of_mediods,:])
    return np.argmin(dists, axis=1)


def dist(xa, xb):
    if EUCLIDEAN:
        return np.sqrt(np.sum(np.square(xa-xb), axis=-1))
    else:
        return np.sum(np.abs(xa - xb), axis=-1)


def find_medoids(assignments):
    medoid_ids = np.full(k, -1, dtype=int)
    subset = np.random.choice(n, batch_sz, replace=False)

    for i in range(k):
        indices = np.intersect1d(np.where(assignments==i)[0], subset)
        distances = dist(x[indices, None, :], x[None, indices, :]).sum(axis=0)
        medoid_ids[i] = indices[np.argmin(distances)]

    return medoid_ids


def kmeds(iterations=20):
    print("Initializing to random medoids.")
    ids_of_medoids = np.random.choice(n, k, replace=False)
    class_assignments = assign_nearest(ids_of_medoids)

    for i in range(iterations):
        print("\tFinding new medoids.")
        ids_of_medoids = find_medoids(class_assignments)
        print("\tReassigning points.")
        new_class_assignments = assign_nearest(ids_of_medoids)

        diffs = np.mean(new_class_assignments != class_assignments)
        class_assignments = new_class_assignments

        print("iteration {:2d}: {:.2%} of points got reassigned."
              "".format(i, diffs))
        if diffs <= 0.01:
            break

    return class_assignments, ids_of_medoids


######################### Generate Fake Data
print("Initializing Data.")
d = 3
k = 6
n = k * 1000000
batch_sz = 1000
x = np.random.normal(size=(n, d))
EUCLIDEAN = False

print("n={}\td={}\tk={}\tbatch_size={} ".format(n, d, k, batch_sz))
print("Distance metric: ", "Eucledian" if EUCLIDEAN else "Manhattan")

print("\nMaking k-groups as:")
for kk in range(k):
    dd = (kk-1)%d
    print("    x[{}:{}, {}] += {}".format(kk*n//k, (kk+1)*n//k, dd , 3*d*kk))
    x[kk*n//k:(kk+1)*n//k,dd] += 3*d*kk

######################### Fitting
print("\nFitting Kmedoids.")
final_assignments, final_medoid_ids = kmeds()

print("\nFitting Kmeans from Scikit-Learn")
fit = KMeans(n_clusters=k).fit(x)
kmeans_assignments = fit.labels_
kmeans = fit.cluster_centers_

mismatch = np.zeros((k, k))
for i, m in (zip(final_assignments, kmeans_assignments)):
    mismatch[i, m] += 1

np.set_printoptions(suppress=True)
print("\nKMedoids:")
print(x[final_medoid_ids, ])
print("K-Medoids class sizes:")
print(mismatch.sum(axis=-1))
print("\nKMeans:")
print(kmeans)
print("K-Means class sizes:")
print(mismatch.sum(axis=0))
print("\nMismatch between assignment to Kmeans and Kmedoids:")
print(mismatch)
print("Should ideally be {} * a permutation matrix.".format(n//k))