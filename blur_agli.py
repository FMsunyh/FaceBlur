import cv2
from numpy import random
import numpy as np


# def get_mask(mask, points, randomize_points=False, random_fraction=0.00):
#     points = np.array(points, dtype=np.int32)
    
#     if len(points) > 2:  # Convex hull requires at least 3 points
#         hull = cv2.convexHull(points)

#         # Fill the convex hull on the mask with white color (255)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # White area is the mask
        
#         if randomize_points:
#             # 找到掩码区域中的所有点
#             mask_points = np.column_stack(np.where(mask == 255))

#             # 根据 random_fraction 确定随机选取的点数
#             num_random_points = int(len(mask_points) * random_fraction)

#             # 使用 numpy.random.choice 进行随机采样
#             chosen_indices = random.choice(len(mask_points), num_random_points, replace=False)
#             random_points = mask_points[chosen_indices]

#             # 在这些随机点上做处理，比如填充不同的颜色
#             for point in random_points:
#                 cv2.circle(mask, (point[1], point[0]), 1, (127, 127, 127), -1)  # 用灰色（127）填充这些点

#     return mask


# def get_mask(mask, points):
#     """
#     根据人脸关键点生成掩码，并对边界进行平滑处理
#     :param mask: 输入的全黑掩码
#     :param points: 人脸关键点
#     :return: 包含平滑边界的人脸区域掩码
#     """
#     points = np.array(points, dtype=np.int32)
    
#     eye1 = np.concatenate((points[17:22, :], points[36:42, :]))
#     if len(points) > 2:  # Convex hull requires至少3个点
#         hull = cv2.convexHull(eye1)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色

#     eye2 = np.concatenate((points[22:27, :], points[42:48, :])) 
#     if len(points) > 2:  # Convex hull requires至少3个点
#         hull = cv2.convexHull(eye2)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
    
#     nose = points[27:36, :]
#     if len(points) > 2:  # Convex hull requires至少3个点
#         hull = cv2.convexHull(nose)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
        
#     nose_eye = np.concatenate((eye1, eye2, nose))
#     if len(points) > 2:  # Convex hull requires至少3个点
#         hull = cv2.convexHull(nose_eye)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
           
#     teeth = points[48:60, :]
#     if len(points) > 2:  # Convex hull requires至少3个点
#         hull = cv2.convexHull(teeth)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
            
#     # 对掩码进行膨胀，扩大边缘区域
#     mask = cv2.dilate(mask, np.ones((35, 35), np.uint8), iterations=1)

#     # 使用高斯模糊平滑掩码边缘
#     mask = cv2.GaussianBlur(mask, (51, 51), 0)

#     return mask



def get_mask(mask, points):
    """
    根据人脸关键点生成掩码，并对边界进行平滑处理
    :param mask: 输入的全黑掩码
    :param points: 人脸关键点
    :return: 包含平滑边界的人脸区域掩码
    """
    points = np.array(points, dtype=np.int32)
    
    # eye1 = np.concatenate((points[17:22, :], points[36:42, :]))
    eye1 = points[36:42, :]
    if len(points) > 2:  # Convex hull requires至少3个点
        hull = cv2.convexHull(eye1)
        cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色

    # eye2 = np.concatenate((points[22:27, :], points[42:48, :])) 
    eye2 = points[42:48, :]
    if len(points) > 2:  # Convex hull requires至少3个点
        hull = cv2.convexHull(eye2)
        cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
    
    nose = points[27:36, :]
    if len(points) > 2:  # Convex hull requires至少3个点
        hull = cv2.convexHull(nose)
        cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
        
    nose_eye = np.concatenate((eye1, eye2, nose))
    if len(points) > 2:  # Convex hull requires至少3个点
        hull = cv2.convexHull(nose_eye)
        cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
           
    teeth = points[48:60, :]
    if len(points) > 2:  # Convex hull requires至少3个点
        hull = cv2.convexHull(teeth)
        cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # 人脸区域为白色
            
    # 对掩码进行膨胀，扩大边缘区域
    mask = cv2.dilate(mask, np.ones((55, 55), np.uint8), iterations=1)

    # 使用高斯模糊平滑掩码边缘
    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    return mask

# def apply_blur(image, landmarks, blur_strength):
#     """
#     对图像的凸包区域应用模糊，使用双边滤波以保留光感和纹理
#     :param image: 输入图像
#     :param landmarks: 人脸特征点
#     :param blur_strength: 模糊强度
#     :return: 处理后的图像
#     """
#     # 获取图像的宽度和高度
#     height, width = image.shape[:2]
    
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
#     # 在掩码上绘制覆盖人脸区域的凸包
#     mask = get_mask(mask, landmarks)
    
#     # 使用双边滤波来模糊图像, 保留边缘和纹理
#     # 参数说明：d表示像素邻域直径，sigmaColor控制颜色空间的过滤，sigmaSpace控制坐标空间的过滤
#     blurred_image = cv2.bilateralFilter(image, d=9, sigmaColor=blur_strength, sigmaSpace=75)
    
#     # 创建结果图像，复制原图
#     result = np.copy(image)
    
#     # 通过掩码将双边滤波后的图像应用到人脸区域
#     result[mask == 255] = blurred_image[mask == 255]
    
#     return result

def apply_strong_blur(image, landmarks, blur_strength=21):
    """
    对图像的人脸区域应用强模糊，使用多次模糊来擦除五官细节
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param blur_strength: 模糊强度，默认值较大以达到五官模糊效果
    :return: 处理后的图像
    """
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 使用特征点生成覆盖人脸区域的掩码
    mask = get_mask(mask, landmarks)
    
    # 高斯模糊 - 用于彻底模糊五官
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # 增加模糊的迭代次数，叠加模糊效果
    for _ in range(10):  # 叠加多次模糊，次数可以根据需求调整
        blurred_image = cv2.GaussianBlur(blurred_image, (blur_strength, blur_strength), 0)
    
    # 双边滤波 - 用于保留轮廓的模糊
    blurred_image = cv2.bilateralFilter(blurred_image, d=9, sigmaColor=blur_strength * 2, sigmaSpace=blur_strength * 2)

    # 创建结果图像，复制原图
    result = np.copy(image)
    
    # 通过掩码将模糊应用到人脸区域
    result[mask == 255] = blurred_image[mask == 255]
    
    return result

def create_soft_mask(mask, kernel_size=51):
    """
    通过模糊掩码创建软边缘掩码
    :param mask: 输入的二值掩码
    :param kernel_size: 高斯核大小，控制边缘的模糊程度
    :return: 带有软边缘的掩码
    """
    # 使用较大的高斯模糊将掩码变为具有软边缘的掩码
    soft_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    # 归一化到0-1的范围，方便后续图像混合
    soft_mask = soft_mask.astype(np.float32) / 255.0
    
    return soft_mask

def apply_natural_blur(image, landmarks, blur_strength=31):
    """
    对人脸区域应用自然的模糊效果，使用软边缘掩码
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param blur_strength: 模糊强度
    :return: 处理后的图像
    """
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 使用特征点生成覆盖人脸区域的掩码
    mask = get_mask(mask, landmarks)
    
    # 创建软边缘掩码，使模糊效果更自然
    soft_mask = create_soft_mask(mask, kernel_size=101)
    
    # 对图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # 创建结果图像，复制原图
    result = np.copy(image)
    
    # 使用软边缘掩码，将模糊效果逐渐过渡到原始图像
    for c in range(3):  # 针对每个颜色通道分别进行操作
        result[:, :, c] = image[:, :, c] * (1 - soft_mask) + blurred_image[:, :, c] * soft_mask

    return result



def high_pass_filter(image, kernel_size=30):
    """
    对图像进行高通滤波，增强边缘
    :param image: 输入图像
    :param kernel_size: 滤波器的尺寸，数值越小高频部分越多
    :return: 经过高通滤波后的图像
    """
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 对图像进行傅里叶变换
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # 将低频信号移到中心
    dft_shift = np.fft.fftshift(dft)
    
    # 获取图像尺寸
    rows, cols = gray_image.shape
    crow, ccol = rows // 2 , cols // 2  # 中心位置
    
    # 创建高通滤波器：中心为零，边缘为1
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-kernel_size:crow+kernel_size, ccol-kernel_size:ccol+kernel_size] = 0
    
    # 应用高通滤波器
    fshift = dft_shift * mask
    
    # 逆变换回图像空间
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # 归一化到0-255的范围
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    
    # 将灰度图转回三通道图像
    img_back = np.uint8(img_back)
    img_back = cv2.merge([img_back, img_back, img_back])
    
    return img_back

def apply_high_pass_with_mask(image, landmarks, blur_strength=31, kernel_size=30):
    """
    在图像上应用高通滤波和模糊，并限定在人脸区域进行处理
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param blur_strength: 高斯模糊的强度
    :param kernel_size: 高通滤波器的尺寸
    :return: 处理后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 根据人脸关键点生成掩码
    mask = get_mask(mask, landmarks)
    
    # 模糊处理
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # 高通滤波处理
    high_pass_image = high_pass_filter(blurred_image, kernel_size)
    
    # 通过掩码将模糊和高通滤波应用在人脸区域
    result = np.copy(image)
    result[mask == 255] = cv2.addWeighted(image[mask == 255], 0.5, high_pass_image[mask == 255], 0.5, 0)
    
    return result

def distance_map(mask):
    """
    计算每个像素点到人脸掩码区域的距离
    :param mask: 人脸区域的掩码
    :return: 距离图，每个像素点到最近的人脸区域的距离
    """
    # 反转掩码，即人脸区域为0，背景为1
    inverted_mask = cv2.bitwise_not(mask)
    
    # 计算每个像素到人脸区域的距离
    dist_map = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    
    # 归一化距离到0-1之间
    dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
    
    return dist_map

def apply_pixelwise_blur(image, landmarks, blur_strength=31):
    """
    根据每个像素点与人脸区域的距离，按像素点应用不同的模糊强度
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param blur_strength: 最大的模糊强度
    :return: 处理后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 根据人脸关键点生成掩码
    mask = get_mask(mask, landmarks)
    
    # 计算距离图
    dist_map = distance_map(mask)
    
    # 创建一个空白图像，用于存储模糊后的结果
    result = np.copy(image)
    
    # 对每个像素应用不同模糊强度
    for y in range(height):
        for x in range(width):
            # 根据距离决定模糊核的大小
            distance_factor = dist_map[y, x]
            blur_kernel_size = int((1 - distance_factor) * blur_strength)
            
            # 确保模糊核大小为奇数且大于1
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if blur_kernel_size < 1:
                blur_kernel_size = 1
            
            # 对局部区域进行模糊处理
            region = image[max(0, y-blur_kernel_size//2):min(height, y+blur_kernel_size//2+1),
                           max(0, x-blur_kernel_size//2):min(width, x+blur_kernel_size//2+1)]
            blurred_pixel = cv2.GaussianBlur(region, (blur_kernel_size, blur_kernel_size), 0)
            
            # 将模糊后的像素点赋值到结果图像中
            result[y, x] = blurred_pixel[blur_kernel_size//2, blur_kernel_size//2]
    
    return result


def smudge_effect(image, landmarks, smudge_radius=15):
    """
    对人脸区域应用涂抹效果，使五官模糊失去细节
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param smudge_radius: 涂抹的半径，控制涂抹程度
    :return: 处理后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 根据人脸关键点生成掩码
    mask = get_mask(mask, landmarks)
    
    # 创建一个副本图像用于涂抹
    result = np.copy(image)
    
    # 对人脸区域应用涂抹效果
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 255:  # 只处理掩码区域内的人脸部分
                # 随机获取涂抹半径范围内的随机位置的像素
                random_x = random.randint(max(0, x - smudge_radius), min(width - 1, x + smudge_radius))
                random_y = random.randint(max(0, y - smudge_radius), min(height - 1, y + smudge_radius))
                
                # 将该随机位置的像素值赋给当前像素
                result[y, x] = image[random_y, random_x]
    
    return result

def smudge_effect2(image, landmarks, smudge_radius=25, iterations=3):
    """
    对人脸区域应用涂抹效果，使五官模糊失去细节，重复多次增强效果
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param smudge_radius: 涂抹的半径，控制涂抹程度
    :param iterations: 涂抹的迭代次数，控制涂抹强度
    :return: 处理后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 根据人脸关键点生成掩码
    mask = get_mask(mask, landmarks)
    
    # 创建一个副本图像用于涂抹
    result = np.copy(image)
    
    # 多次涂抹，增强模糊效果
    for _ in range(iterations):
        # 对人脸区域应用涂抹效果
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 255:  # 只处理掩码区域内的人脸部分
                    # 随机获取涂抹半径范围内的随机位置的像素
                    random_x = random.randint(max(0, x - smudge_radius), min(width - 1, x + smudge_radius))
                    random_y = random.randint(max(0, y - smudge_radius), min(height - 1, y + smudge_radius))
                    
                    # 将该随机位置的像素值赋给当前像素
                    result[y, x] = image[random_y, random_x]
    
    return result



def smudge_effect3(image, landmarks, smudge_radius=25, iterations=3):
    """
    对人脸区域应用涂抹效果，使五官模糊失去细节，同时边界过渡自然
    :param image: 输入图像
    :param landmarks: 人脸特征点
    :param smudge_radius: 涂抹的半径，控制涂抹程度
    :param iterations: 涂抹的迭代次数，控制涂抹强度
    :return: 处理后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个全黑的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 根据人脸关键点生成掩码，并对边界进行平滑
    mask = get_mask(mask, landmarks)
    
    # 创建一个副本图像用于涂抹
    result = np.copy(image)
    
    # 多次涂抹，增强模糊效果
    for _ in range(iterations):
        # 对人脸区域应用涂抹效果
        for y in range(height):
            for x in range(width):
                # 只对掩码区域和边缘平滑区域进行处理
                if mask[y, x] > 0:
                    # 根据掩码的灰度值决定模糊强度，值越大，模糊越强
                    alpha = mask[y, x] / 255.0
                    
                    # 随机获取涂抹半径范围内的随机位置的像素
                    random_x = random.randint(max(0, x - smudge_radius), min(width - 1, x + smudge_radius))
                    random_y = random.randint(max(0, y - smudge_radius), min(height - 1, y + smudge_radius))
                    
                    # 将该随机位置的像素值与当前像素值按alpha混合，产生自然过渡
                    result[y, x] = (1 - alpha) * image[y, x] + alpha * image[random_y, random_x]
    
    return result


def average_color_in_radius(image, x, y, radius):
    """
    计算给定半径内像素的平均颜色，用于生成模糊效果
    :param image: 输入图像
    :param x: 像素点的x坐标
    :param y: 像素点的y坐标
    :param radius: 模糊半径
    :return: 邻域内的平均颜色
    """
    height, width = image.shape[:2]
    x_min = max(0, x - radius)
    x_max = min(width, x + radius)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius)
    
    # 提取区域并计算平均颜色
    region = image[y_min:y_max, x_min:x_max]
    avg_color = np.mean(region, axis=(0, 1))  # 计算区域内所有像素的平均值
    
    return avg_color

# def smudge_effect4(image, landmarks, smudge_radius=25, iterations=15):
#     """
#     对人脸区域应用涂抹效果，使五官模糊失去细节，同时边界过渡自然
#     :param image: 输入图像
#     :param landmarks: 人脸特征点
#     :param smudge_radius: 涂抹的半径，控制涂抹程度
#     :param iterations: 涂抹的迭代次数，控制涂抹强度
#     :return: 处理后的图像
#     """
#     # 获取图像尺寸
#     height, width = image.shape[:2]
    
#     # 创建一个全黑的掩码
#     mask = np.zeros((height, width), dtype=np.uint8)
    
#     # 根据人脸关键点生成掩码，并对边界进行平滑
#     mask = get_mask(mask, landmarks)
    
#     # 创建一个副本图像用于涂抹
#     result = np.copy(image)
    
#     # 多次涂抹，增强模糊效果
#     for _ in range(iterations):
#         # 对人脸区域应用涂抹效果
#         for y in range(height):
#             for x in range(width):
#                 # 只对掩码区域和边缘平滑区域进行处理
#                 if mask[y, x] > 0:
#                     # 根据掩码的灰度值决定模糊强度，值越大，模糊越强
#                     alpha = mask[y, x] / 255.0
                    
#                     # 使用邻域的平均颜色代替随机像素，确保过渡更自然
#                     avg_color = average_color_in_radius(image, x, y, smudge_radius)
                    
#                     # 按alpha混合当前像素和模糊后的像素，产生自然过渡
#                     result[y, x] = (1 - alpha) * image[y, x] + alpha * avg_color
    
#     return result


# def smudge_effect5(image, landmarks, smudge_radius=25, iterations=10):
#     """
#     对人脸区域应用涂抹效果，使五官模糊失去细节，同时边界过渡自然
#     :param image: 输入图像
#     :param landmarks: 人脸特征点
#     :param smudge_radius: 涂抹的半径，控制涂抹程度
#     :param iterations: 涂抹的迭代次数，控制涂抹强度
#     :return: 处理后的图像
#     """
#     # 获取图像尺寸
#     height, width = image.shape[:2]
    
#     # 创建一个全黑的掩码
#     mask = np.zeros((height, width), dtype=np.uint8)
    
#     # 根据人脸关键点生成掩码，并对边界进行平滑
#     mask = get_mask(mask, landmarks)
    
#     # 创建一个副本图像用于涂抹
#     result = np.copy(image)
    
#     # 多次涂抹，增强模糊效果
#     for _ in range(iterations):
#         # 对人脸区域应用涂抹效果
#         for y in range(height):
#             for x in range(width):
#                 # 只对掩码区域和边缘平滑区域进行处理
#                 if mask[y, x] > 0:
#                     # 根据掩码的灰度值决定模糊强度，值越大，模糊越强
#                     alpha = mask[y, x] / 255.0
                    
#                     # 使用邻域的平均颜色代替随机像素，确保过渡更自然
#                     avg_color = average_color_in_radius(image, x, y, smudge_radius)
                    
#                     # 计算距离，衰减效果
#                     distance = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
#                     distance_factor = max(0, 1 - distance / (width / 2))  # 距离因子
#                     distance_factor = distance_factor ** 2  # 增强衰减效果
                    
#                     # 按alpha和距离因子混合当前像素和模糊后的像素，产生更加柔和的过渡
#                     result[y, x] = (1 - alpha * distance_factor) * image[y, x] + alpha * distance_factor * avg_color
    
#     return result

def add_random_fog(image, mask, fog_intensity=0.5):
    height, width = image.shape[:2]

    # 创建随机噪声图像
    noise = np.random.rand(height, width, 3) * 255
    noise = noise.astype(np.uint8)

    # 创建雾化效果图像
    fogged_image = cv2.addWeighted(image, 1 - fog_intensity, noise, fog_intensity, 0)

    # 仅在掩码区域内应用雾化效果
    fogged_image = np.where(mask[:, :, np.newaxis] > 0, fogged_image, image)

    return fogged_image

# def smudge_effect6(image, landmarks, smudge_radius=25, iterations=10):
#     """
#     对人脸区域应用涂抹效果，使五官模糊失去细节
#     """
#     height, width = image.shape[:2]
#     mask = np.zeros((height, width), dtype=np.uint8)

#     mask = get_mask(mask, landmarks)
#     result = np.copy(image)

#     # 使用多次迭代加强模糊效果
#     for _ in range(iterations):
#         for y in range(height):
#             for x in range(width):
#                 if mask[y, x] > 0:  # 只处理掩码区域
#                     avg_color = average_color_in_radius(image, x, y, smudge_radius)
#                     result[y, x] = avg_color

#     # 添加边缘过渡效果
#     blurred_mask = cv2.GaussianBlur(mask, (31, 31), 0)
#     for y in range(height):
#         for x in range(width):
#             if blurred_mask[y, x] > 0:
#                 alpha = blurred_mask[y, x] / 255.0
#                 result[y, x] = (1 - alpha) * image[y, x] + alpha * result[y, x]

#     return result


# def smudge_effect7(image, landmarks, smudge_radius=25, iterations=11):
#     """
#     对图像的圆角矩形区域应用模糊
#     :param image: 输入图像
#     :param radius: 圆角半径
#     :return: 处理后的图像
#     """
#     # 获取图像的宽度和高度
#     blurratio=[3,9,15,21,31,41,51,61,71,81,101]
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
#     # 在掩码上绘制覆盖整个图像的圆角矩形
#     mask = get_mask(mask, landmarks)
#     blurred_image = image

#     # 对图像进行模糊处理
#     for i in range(iterations):
#         blurred_image = cv2.GaussianBlur(blurred_image, (blurratio[iterations-i-1], blurratio[iterations-i-1]), 0)

#     # 创建结果图像，复制原图
#     result = np.copy(image)

#     # 通过掩码将模糊应用到圆角矩形区域
#     result[mask == 255] = blurred_image[mask == 255]

#     return result

def smudge_effect8(image, landmarks, smudge_radius=50, iterations=10, fog_intensity=0.5):
    """
    对人脸区域应用涂抹效果并添加雾化效果
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    mask = get_mask(mask, landmarks)
    result = np.copy(image)

    # 迭代涂抹效果
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    avg_color = average_color_in_radius(image, x, y, smudge_radius)
                    result[y, x] = avg_color

    # 添加边缘过渡效果
    blurred_mask = cv2.GaussianBlur(mask, (51, 51), 0)
    for y in range(height):
        for x in range(width):
            if blurred_mask[y, x] > 0:
                alpha = blurred_mask[y, x] / 255.0
                result[y, x] = (1 - alpha) * image[y, x] + alpha * result[y, x]

    # 添加随机性雾化效果
    result = add_random_fog(result, mask, fog_intensity)

    return result


def smudge_effect9(image, landmarks, avg_color=0, smudge_radius=50, iterations=10, fog_intensity=0.5):
    """
    对人脸区域应用涂抹效果并添加雾化效果
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    mask = get_mask(mask, landmarks)
    result = np.copy(image)

    # 迭代涂抹效果
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    result[y, x] = avg_color

    # 添加边缘过渡效果
    blurred_mask = cv2.GaussianBlur(mask, (51, 51), 0)
    for y in range(height):
        for x in range(width):
            if blurred_mask[y, x] > 0:
                alpha = blurred_mask[y, x] / 255.0
                result[y, x] = (1 - alpha) * image[y, x] + alpha * result[y, x]

    # 添加随机性雾化效果
    result = add_random_fog(result, mask, fog_intensity)

    return result

apply_blur = smudge_effect9