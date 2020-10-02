import numpy as np
import time

from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql import functions as F
from sklearn.neighbors import NearestNeighbors
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Constants
N = 100000  # number of random points in the dataset
num_tests = 1  # number of test iterations
dim = 3  # number of dimensions of the points
noise_sigma = .01  # standard deviation error to be added
translation = .1  # max translation of the test set
rotation = .1  # max rotation (radians) of the test set


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    #    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        print("iteration #" + str(i))
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def read_ply(path):
    # reading the text file from the location provided in the arguments
    txt = spark.read.text(path)

    # removing the header of the file
    header = spark.createDataFrame(txt.head(17))

    # getting the difference of the two dataframe, thus getting only the datapoints
    df = txt.subtract(header)

    # splitting the row into columns, defining their types and removing the string column
    final = df \
        .withColumn("x", F.split("value", " ").getItem(0).cast(FloatType())) \
        .withColumn("y", F.split("value", " ").getItem(1).cast(FloatType())) \
        .withColumn("z", F.split("value", " ").getItem(2).cast(FloatType())) \
        .withColumn("nx", F.split("value", " ").getItem(3).cast(FloatType())) \
        .withColumn("ny", F.split("value", " ").getItem(4).cast(FloatType())) \
        .withColumn("nz", F.split("value", " ").getItem(5).cast(FloatType())) \
        .withColumn("red", F.split("value", " ").getItem(6).cast(IntegerType())) \
        .withColumn("green", F.split("value", " ").getItem(7).cast(IntegerType())) \
        .withColumn("blue", F.split("value", " ").getItem(8).cast(IntegerType())) \
        .withColumn("alpha", F.split("value", " ").getItem(9).cast(IntegerType())) \
        .drop("value")

    # geting the dataframe as Numpy array
    POINT_CLOUD = np.array(final.collect())

    return POINT_CLOUD


def read_ply_from_table(tablename):
    # reading the table  from the location provided in the arguments
    df = spark.sql("select * from newtable")

    # geting the dataframe as Numpy array
    POINT_CLOUD = np.array(df.collect())

    return POINT_CLOUD


def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def run_icp():
    A = read_ply('room_full.ply')
    total_time = 0

    for i in range(num_tests):
        B = read_ply('room_full_quarter_rotated.ply')

        # Run ICP
        start = time.time()
        T, distances, iterations = icp(B, A, max_iterations=10000, tolerance=0.00001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((len(B), 4))
        C[:, 0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

    print('icp time: {:.3}'.format(total_time / num_tests))

    return C, T


def import_packages(x):
    import numpy as np
    import time
    from sklearn.neighbors import NearestNeighbors
    return x


if __name__ == "__main__":
    sc = SparkContext(appName="test")
    spark = SparkSession.builder.appName("PLYtoTable").getOrCreate()
    int_rdd = sc.parallelize([1, 2, 3, 4])
    int_rdd.map(lambda x: import_packages(x))
    int_rdd.collect()
    C, T = run_icp()
    print(T)
