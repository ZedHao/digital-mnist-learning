from joblib import dump, load
import cv2
import numpy as np
import numpy as np
from PIL import Image
import os
import re
import pdb
def preprocess_image(image_path):
    # 读取图像并将其转换为灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}（路径错误或文件损坏）")

    # 对图像进行阈值处理，将其转换为二值图像
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # 找到图像的边界框
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pdb.set_trace()
    x, y, w, h = cv2.boundingRect(contours[0])

    # 将图像调整为MNIST数据集中的标准尺寸（28x28像素）
    digit_image = binary_image[y:y+h, x:x+w]
    resized_image = cv2.resize(digit_image, (28, 28))

    # 将像素值缩放到0到1之间
    scaled_image = resized_image / 255.0

    # 将图像转换为MNIST数据集中的格式
    mnist_format_image = scaled_image.reshape(1, 28*28)

    return mnist_format_image


def preprocess_image_v2(image_path):


    # 读取训练好的模型

    # 读取图片转成灰度格式
    img = Image.open(image_path).convert('L')

    # resize的过程
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))

    # 暂存像素值的一维数组
    arr = []

    for i in range(28):
        for j in range(28):
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            pixel = 1.0 - float(img.getpixel((j, i)))/255.0
            # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
            arr.append(pixel)

    arr1 = np.array(arr).reshape((1, 28*28))
    return arr1




def extract_numbers_from_image_filenames(folder_path, pkl_path):
    # 初始化一个空列表，用于存储提取出的数字
    numbers = []
    if not os.path.exists(folder_path):
        raise NotADirectoryError(f"文件夹不存在：{folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} 不是一个文件夹")
    for filename in os.listdir(folder_path):

        # 检查文件是否以 "num" 开头并且以 ".png" 结尾
        if filename.startswith("num") and filename.endswith(".png"):
            # 使用正则表达式从文件名中提取数字部分
            number = re.findall(r'\d+', filename)
            if number:  # 确保找到了数字
                check_data("./algorithm/image/num{idx}.png".format(idx=number[0]), pkl_path)

    return numbers






def check_data(file_name,pkl_path):
    # 示例：将图像转换为MNIST格式
    image_path = file_name
    mnist_image = preprocess_image_v2(image_path)

    loaded_model = load( pkl_path)
    # 使用加载后的模型进行预测
    predictions = loaded_model.predict(mnist_image)
    print(file_name, "fileName---Predicted Digit:", predictions[0])


if __name__ == '__main__':
    extract_numbers_from_image_filenames("./algorithm/image","./algorithm/saved_model/lr.pkl")
