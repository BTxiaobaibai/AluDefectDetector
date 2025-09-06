# ==============================================================================
# 铝材缺陷数据集准备 (The Ultimate Dataset Preparation Script)
# ##自动完成清理、创建目录、转换、划分、复制所有步骤##
#
# **脚本核心使命**:
#   一步到位，将原始的、可能存在问题的COCO格式数据集，
#   转换并严格划分为一个干净、标准、无重复、可直接用于YOLOv5训练的数据集。
#   这个脚本是整个数据准备阶段的唯一权威。
#
# **执行逻辑**:
#   1. 清理旧的输出，保证每次运行都是全新的开始。
#   2. 创建YOLOv5标准的目录结构 (images/train, images/val, labels/train, labels/val)。
#   3. **只读取 train.json** 作为唯一的、权威的标注来源，以从根本上避免重复标注问题。
#   4. 在内存中，将所有COCO标注信息按图片进行分组和格式转换。
#   5. 对**所有带标注的图片**进行随机、严格的80/20划分。
#   6. 根据划分好的名单，将对应的图片和标签文件复制到最终的目标文件夹。
# ==============================================================================


# --- 侦探的工具箱：导入所有需要的Python库 ---
import json  # 用于解析JSON格式的“案件记录本”
import os  # 用于处理文件和文件夹路径，让代码能在不同操作系统上运行
import random  # 用于随机打乱数据，这是保证模型泛化能力的关键
import shutil  # 高级文件操作工具箱，用于删除整个文件夹和复制文件
from tqdm import tqdm  # 一个非常友好的进度条库，让我们能直观地看到脚本的执行进度

# --- 1. 全局设定：定义所有路径和关键参数 ---
# 在这里统一定义所有变量，是一种专业的编程习惯，便于后期修改和维护。

# **“案发现场”的根目录**：我们所有的数据文件都存放在这里面。
# 使用 `r"..."` 原始字符串格式，可以防止Windows路径中的反斜杠 `\` 被误解。
base_dir = r"D:\Storage\Backend_project_python\school\AluDefectDetector\dataset\aluminum"

# **“原始证物”的位置**：
# source_images_dir 指向存放全部412张原始图片的文件夹。
source_images_dir = os.path.join(base_dir, "images")
# coco_annotation_path 指向我们选定的唯一、权威的标注源文件。
# 我们选择 train.json，因为它通常包含了大部分甚至全部的标注信息。
coco_annotation_path = os.path.join(base_dir, "annotations", "train.json")

# **“最终归档”的位置**：
# output_dataset_dir 定义了我们将要创建的、存放所有最终成果的文件夹名。
output_dataset_dir = os.path.join(base_dir, "yolo_dataset")

# **“游戏规则”的设定**：
# VALIDATION_SPLIT 定义了验证集所占的比例。0.2意味着20%的数据用于“模拟考试”。
VALIDATION_SPLIT = 0.2
# RANDOM_SEED 是一个随机种子。把它想象成一副扑克牌的特定洗牌手法。
# 只要这个数字不变，每次运行脚本，“洗牌”（随机打乱）的结果都将完全一样。
# 这对于科学研究和工程调试至关重要，因为它保证了实验的“可复现性”。
RANDOM_SEED = 42

# --- 2. 安全措施：施工前，必须彻底清场 ---
# 这一步是为了解决“重复运行”可能导致的数据污染问题。

# os.path.exists() 检查一个文件或文件夹是否存在。
if os.path.exists(output_dataset_dir):
    # 如果我们的目标文件夹已经存在，说明可能残留着上次运行的结果。
    print(f"警告：发现旧的输出目录 '{output_dataset_dir}'，正在进行彻底清理...")
    # shutil.rmtree() 是一个强力删除命令，它会删除整个文件夹及其内部的所有内容。
    shutil.rmtree(output_dataset_dir)
    print("旧目录清理完成。")

# --- 3. 搭建全新的、标准的“档案室”结构 ---
print("正在创建新的YOLO标准目录结构...")
# 我们需要为训练和验证的图片、标签分别创建独立的文件夹。
output_images_train_dir = os.path.join(output_dataset_dir, "images", "train")
output_images_valid_dir = os.path.join(output_dataset_dir, "images", "val")
output_labels_train_dir = os.path.join(output_dataset_dir, "labels", "train")
output_labels_valid_dir = os.path.join(output_dataset_dir, "labels", "val")
# 将所有需要创建的路径放入一个列表，方便用循环处理。
for path in [output_images_train_dir, output_images_valid_dir, output_labels_train_dir, output_labels_valid_dir]:
    # os.makedirs() 会创建指定的文件夹。如果路径中的父文件夹（如'images'）不存在，它也会一并创建。
    os.makedirs(path)
print("目录创建完毕，结构清晰，准备就绪！")


# --- 4. 核心执行函数：一步到位完成所有数据处理 ---
def process_dataset():
    """
    这是整个脚本的“主引擎”，它封装了所有的核心逻辑。
    """

    # --- 4.1. 加载并解析唯一的“案件记录本” (COCO JSON) ---
    print(f"\n步骤 1/4: 正在从唯一的源文件 '{os.path.basename(coco_annotation_path)}' 加载标注信息...")
    with open(coco_annotation_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 将JSON中的主要部分提取到独立的变量中，便于后续处理。
    images_info = coco_data['images']
    annotations_info = coco_data['annotations']
    print("标注信息加载并解析完成。")

    # --- 4.2. 在内存中，将所有标注信息进行转换和分组 ---
    # 目标：创建一个字典，结构为 {'图片名': ['yolo格式标注1', 'yolo格式标注2', ...]}
    # 这样做的好处是，所有计算都在内存中完成，最后再一次性写入文件，效率高且逻辑清晰。

    print(f"\n步骤 2/4: 正在将所有COCO标注转换为YOLO格式并按图片分组...")

    # 首先，创建一个图片ID到图片信息的“速查手册”，提高效率。
    image_id_map = {img['id']: img for img in images_info}

    # 创建我们的核心数据结构：一个用于存放按图片分组后标注的字典。
    annotations_by_image = {}

    # 使用tqdm包装annotations_info列表，以显示处理进度条。
    for ann in tqdm(annotations_info):
        # 提取当前这条标注的关键信息
        image_id = ann['image_id']
        category_id = ann['category_id']
        coco_bbox = ann['bbox']

        # 通过图片ID，从“速查手册”中查询图片详情。
        img_details = image_id_map.get(image_id)
        if not img_details:  # 如果查不到（数据可能存在问题），则跳过这条标注。
            continue

        # 获取图片原始宽高，这是归一化计算的“基准”。
        img_w = img_details['width']
        img_h = img_details['height']

        # --- 执行从COCO到YOLO的核心数学转换 ---
        coco_x, coco_y, coco_w, coco_h = coco_bbox
        x_center = coco_x + coco_w / 2
        y_center = coco_y + coco_h / 2
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = coco_w / img_w
        height_norm = coco_h / img_h
        yolo_category_id = category_id - 1

        # 将计算结果格式化为YOLO标准的字符串，保留6位小数是通用惯例。
        yolo_line = f"{yolo_category_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

        # 获取这张标注对应的图片文件名。
        image_filename = os.path.basename(img_details['file_name'])

        # --- 将转换好的标注“存入”我们的大字典中 ---
        # 检查这个图片名是否已经是字典的键了。
        if image_filename not in annotations_by_image:
            # 如果是第一次遇到这张图片，就在字典里为它创建一个空列表作为值。
            annotations_by_image[image_filename] = []
        # 将当前这条转换好的YOLO标注字符串，添加到对应图片名的列表中。
        annotations_by_image[image_filename].append(yolo_line)

    print("所有标注已在内存中转换并分组完毕。")

    # --- 4.3. 核心步骤：随机、严格地划分图片名单 ---
    print(f"\n步骤 3/4: 正在对所有带标注的图片进行随机划分...")

    # annotations_by_image.keys() 会返回所有有标注的图片的文件名。
    annotated_images = list(annotations_by_image.keys())

    # 设定随机种子，保证每次划分结果一致。
    random.seed(RANDOM_SEED)
    # 彻底打乱文件名列表。
    random.shuffle(annotated_images)

    # 计算80%训练集的分割点。
    split_point = int(len(annotated_images) * (1 - VALIDATION_SPLIT))

    # 使用列表切片，将文件名列表分割成两个互不重叠的子列表。
    train_files = annotated_images[:split_point]
    valid_files = annotated_images[split_point:]

    print(f"\n划分结果:")
    print(f"  - 总计有标注的图片数: {len(annotated_images)}")
    print(f"  - 训练集图片数 (80%): {len(train_files)}")
    print(f"  - 验证集图片数 (20%): {len(valid_files)}")

    # --- 4.4. 按划分好的名单，将文件分发到最终的“档案室” ---
    print("\n步骤 4/4: 正在将图片和标签文件复制到最终目录...")

    # (这是一个内部辅助函数，用于避免重复代码)
    def distribute_files(filenames, image_dest_dir, label_dest_dir):
        for filename in tqdm(filenames):
            # 拼接源文件路径
            source_image_path = os.path.join(source_images_dir, filename)

            # 拼接目标文件路径
            target_image_path = os.path.join(image_dest_dir, filename)
            target_label_path = os.path.join(label_dest_dir, os.path.splitext(filename)[0] + ".txt")

            # 复制图片文件
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, target_image_path)

            # 从内存中的大字典里，获取这张图片的所有YOLO标注
            yolo_content = annotations_by_image[filename]
            # 一次性将所有标注写入新的.txt文件
            with open(target_label_path, 'w', encoding='utf-8') as f:
                # '\n'.join(...) 是一个高效的写法，它会用换行符把列表里的所有字符串连接起来。
                f.write('\n'.join(yolo_content))

    # 分发训练集
    print(" -> 正在分发训练集...")
    distribute_files(train_files, output_images_train_dir, output_labels_train_dir)

    # 分发验证集
    print(" -> 正在分发验证集...")
    distribute_files(valid_files, output_images_valid_dir, output_labels_valid_dir)

    print("\n==================================================")
    print("恭喜！终极数据集已准备完毕，随时可以开始训练！")
    print("==================================================")


# --- 主程序入口：当直接运行此脚本时，执行核心函数 ---
if __name__ == "__main__":
    process_dataset()