import os
import shutil
import random
from dotenv import load_dotenv
load_dotenv()

moca_dir = os.environ.get('MOCA_DIR')
fu3_dir = os.environ.get('FU3_DIR')
fu4_dir = os.environ.get('FU4_DIR')
target = os.environ.get('TARGET')

# create new folder structure
os.makedirs(target, exist_ok=True)
splits = ["train", "val", "test"]
subfolder = ["A", "B"]
for split in splits:
    for name in subfolder:
        os.makedirs(os.path.join(target, split, name), exist_ok=True)

remove_files = ["1546_FU3_20160428_CDT_extracted.png", 
                "7675_FU3_20160428_CDT_extracted.png", 
                "1121_FU3_20160422_CDT_extracted.png", 
                "7575_FU3_20160422_CDT_extracted.png", 
                "9087_FU3_20160422_CDT_extracted.png",
                "1546_FU3_20160428_CDT_extracted.png",
                "7675_FU3_20160428_CDT_extracted.png",
                "1121_FU3_20160422_CDT_extracted.png",
                "7575_FU3_20160422_CDT_extracted.png",
                "9087_FU3_20160422_CDT_extracted.png",
                "7108_FU3_20160421_CDT_extracted.png",
                "7048_FU3_20160512_CDT_extracted.png",
                "7314_FU3_20160512_CDT_extracted.png"]      

moca_files = [os.path.join(moca_dir, f) for f in os.listdir(moca_dir) if f.lower().endswith(".png")]
static_files = [os.path.join(fu3_dir, f) for f in os.listdir(fu3_dir)]
static_files += [os.path.join(fu4_dir, f) 
             for f in os.listdir(fu4_dir) 
             if f not in remove_files]

random.shuffle(static_files)
random.shuffle(moca_files)

def get_splits(files):
    train_idx = int(len(files) * 0.7)
    val_idx = int(len(files) * 0.9)
    train = files[:train_idx]
    val = files[train_idx:val_idx]
    test = files[val_idx:]
    return train, val, test

train_moca, val_moca, test_moca = get_splits(moca_files)
train_static, val_static, test_static = get_splits(static_files)


# copy digital drawings in B
for src in train_moca:
    dst = os.path.join(target, "train", "B", os.path.basename(src))
    shutil.copy2(src, dst)

for src in val_moca:
    dst = os.path.join(target, "val", "B", os.path.basename(src))
    shutil.copy2(src, dst)

for src in test_moca:
    dst = os.path.join(target, "test", "B", os.path.basename(src))
    shutil.copy2(src, dst)

# copy scanned drawings in A
for src in train_static:
    dst = os.path.join(target, "train", "A", os.path.basename(src))
    shutil.copy2(src, dst)

for src in val_static:
    dst = os.path.join(target, "val", "A", os.path.basename(src))
    shutil.copy2(src, dst)

for src in test_static:
    dst = os.path.join(target, "test", "A", os.path.basename(src))
    shutil.copy2(src, dst)

def check_overlap(dir_1, dir_2):
    files_1 = set(os.listdir(dir_1))
    files_2 = set(os.listdir(dir_2))
    
    overlap = files_1 & files_2 
    
    if overlap:
        print(f"overlap detected ({len(overlap)} files) between {dir_1} and {dir_2}:")
        for f in overlap:
            print(f)
        return False
    else:
        print(f"no overlaps between {dir_1} and {dir_2}")
        return True

train_A_dir = os.path.join(target, "train", "A")
train_B_dir = os.path.join(target, "train", "B")
val_A_dir = os.path.join(target, "val", "A")
val_B_dir = os.path.join(target, "val", "B")
test_A_dir = os.path.join(target, "test", "A")
test_B_dir = os.path.join(target, "test", "B")

check_overlap(train_A_dir, val_A_dir)
check_overlap(train_A_dir, test_A_dir)
check_overlap(val_A_dir, test_A_dir)

check_overlap(train_B_dir, val_B_dir)
check_overlap(train_B_dir, test_B_dir)
check_overlap(val_B_dir, test_B_dir)







    