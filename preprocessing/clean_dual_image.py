import os
from PIL import Image
import imagehash

# CONFIG
base_path = "/Users/Ding/Desktop/NUS-proj/mdata_new/train/Ragdolls"
folder_list = os.listdir(base_path)
delete = True  # Set to False if you just want to preview duplicates
hash_size = 16  # Larger size = more sensitive to detail
threshold = 0   # Hamming distance: 0 = exact duplicate, 5 = visually similar

# Supported image extensions
img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

hash_dict = {}
global index 
index = 0

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in img_exts)

def clean_duplicates(path):
    global index
    for root, _, files in os.walk(path):
        for filename in files:
            if index > 500: break
            if not is_image_file(filename):
                continue

            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as img:
                    img_hash = imagehash.phash(img, hash_size=hash_size)

                for stored_hash, stored_path in hash_dict.items():
                    if img_hash - stored_hash <= threshold:
                        print(f"Duplicate found: {filepath} ≈ {stored_path}")
                        index = index + 1
                        if delete:
                            os.remove(filepath)
                            
                            print(f"Deleted: {filepath}")
                        break
                else:
                    hash_dict[img_hash] = filepath

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    # for i in range(len(folder_list)):
    # folder_path = os.path.join(base_path, folder_list[i])
    folder_path = base_path
    if not os.path.exists(folder_path): 
        print("File Folder not found!")
    else:
        print("Cleaning Folder: %s"%folder_path)
        clean_duplicates(folder_path)
    print(index)
