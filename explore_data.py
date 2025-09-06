# 作用：用于随机抽取train中的数据，再在images中找到，基于train中抽取的数据渲染标注和框(目的：检测是否能正常拿到数据并展示图片)
import json
import os
import random
import cv2

# --- 1. 配置你的文件路径 ---
#    请确保这里的路径和你电脑上的实际路径一致

# 数据集的根目录
base_dir = r"D:\Storage\Backend_project_python\school\AluDefectDetector\dataset\aluminum"

# 图片文件夹的完整路径
image_dir = os.path.join(base_dir, "images")

# 标注文件的完整路径
annotation_path = os.path.join(base_dir, "annotations", "train.json")


# --- 2. 加载并解析JSON标注文件 ---

print(f"正在从 {annotation_path} 加载标注文件...")
with open(annotation_path, 'r') as f:
    # json.load() 会把json文件内容读取成一个Python字典
    annotations_data = json.load(f)
    print(annotations_data.keys())
print("加载完成！")

# 从JSON数据中提取三部分重要信息
images_info = annotations_data['images']
annotations_info = annotations_data['annotations']
categories_info = annotations_data['categories']

# 为了方便查找，我们创建一个从 类别ID -> 类别名称 的映射字典
# 比如 {1: 'scratch', 2: 'pore'}
category_map = {item['id']: item['name'] for item in categories_info}
print(f"检测到 {len(category_map)} 个类别: {category_map}")

# --- 3. 随机选择一个样本并进行可视化 ---

# 从所有标注中随机抽取一个
random_annotation = random.choice(annotations_info)

# 从这个标注中获取 图像ID 和 缺陷类别ID
image_id = random_annotation['image_id']
category_id = random_annotation['category_id']

# 获取边界框(bounding box)信息，格式是 [x, y, width, height]
bbox = random_annotation['bbox']
x, y, w, h = [int(i) for i in bbox] # 转换成整数

# 使用 图像ID 找到对应的图片文件名
image_filename = ""
for img_info in images_info:
    if img_info['id'] == image_id:
        image_filename = os.path.basename(img_info['file_name'])
        break

# --- 4. 使用OpenCV读取图片并绘制 ---

if image_filename:
    # 构造完整的图片路径
    image_path = os.path.join(image_dir, image_filename)
    print(f"\n正在可视化图片: {image_path}")

    # 读取图片
    image = cv2.imread(image_path)

    # 在图片上画出红色的矩形框
    # (255, 0, 0) 代表 BGR 颜色中的红色
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 在框的上方写上类别名称
    category_name = category_map[category_id]
    cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 创建一个窗口并显示图片
    cv2.imshow(f"Data Exploration: {image_filename}", image)

    # 等待用户按任意键后关闭窗口
    print("请按任意键关闭预览窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"错误：找不到ID为 {image_id} 的图片信息！")