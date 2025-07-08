# # This python code serves the following functions:
# 1. Check for standard dataset structure file_folder src_folder (name - train/validation - class - images)
# 2. padding images for class with fewer samples
# 3. create a new balanced dataset at dst_dic

import os
import shutil

src_dir = './mdata'           
dst_dir = './mdata_balanced'  
os.makedirs(dst_dir, exist_ok=True)

for split in ['train', 'test', 'validation']:
    split_src_dir = os.path.join(src_dir, split)
    split_dst_dir = os.path.join(dst_dir, split)
    
    if not os.path.exists(split_src_dir):
        print(f"警告: {split} 文件夹不存在，跳过")
        continue
    
    os.makedirs(split_dst_dir, exist_ok=True)
    
    cat_counts = {}
    cat_imgs = {}
    
    for cat in os.listdir(split_src_dir):
        cat_path = os.path.join(split_src_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        imgs = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) == 0:
            print(f"警告: {split}/{cat} 文件夹为空，跳过")
            continue
        cat_counts[cat] = len(imgs)
        cat_imgs[cat] = imgs
        print(f"{split}/{cat}: {len(imgs)} 张图片")
    
    # 找到最大图片数
    if not cat_counts:
        print(f"错误: {split} 中没有找到任何有效的图片类别")
        continue
    
    max_count = max(cat_counts.values())
    print(f"{split} 最大图片数: {max_count}")
    
    # 为每个类别补齐图片
    for cat, imgs in cat_imgs.items():
        print(f"处理类别: {split}/{cat}")
        src_cat_path = os.path.join(split_src_dir, cat)
        dst_cat_path = os.path.join(split_dst_dir, cat)
        os.makedirs(dst_cat_path, exist_ok=True)
        # 循环补齐到max_count
        for i in range(max_count):
            img_name = imgs[i % len(imgs)]
            src_img = os.path.join(src_cat_path, img_name)
            # 编号格式：类别名_序号.jpg
            dst_img = os.path.join(dst_cat_path, f"{cat}_{i+1:05d}.jpg")
            shutil.copy(src_img, dst_img)
            
        # 输出补齐统计信息
        original_count = len(imgs)
        copies_per_image = max_count // original_count
        remaining = max_count % original_count
        print(f"  原始图片: {original_count} 张")
        print(f"  每张图片复制: {copies_per_image} 次")
        print(f"  前 {remaining} 张图片额外复制: 1 次")

print("所有类别已补齐到相同数量，图片已统一编号。")