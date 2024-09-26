import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
from scipy.spatial import ConvexHull


# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

try:
    input_img = io.imread('../data/alignment/aflw-test.jpg')
except FileNotFoundError:
    input_img = io.imread('data/alignment/aflw-test.jpg')

preds = fa.get_landmarks(input_img)[-1]
points = preds[:,:2]

# 计算凸包
hull = ConvexHull(points)

# 绘制点和凸包
plt.scatter(points[:, 0], points[:, 1], label='Random Points')

# 绘制凸包的边
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

# 标注顶点
for i in hull.vertices:
    plt.text(points[i, 0], points[i, 1], f'{i}', color='blue')

plt.title('Convex Hull of 68 Random Points')
plt.legend()
plt.show()