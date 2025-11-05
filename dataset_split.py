import os
import shutil
import random
from dotenv import load_dotenv
load_dotenv()

moca_dir = os.environ.get('MOCA_DIR')
fu3_dir = os.environ.get('FU3_DIR')
fu4_dir = os.environ.get('FU4_DIR')
target = os.environ.get('TARGET')
print(moca_dir, fu3_dir, fu4_dir, target)

# create new folder structure
os.makedirs(target, exist_ok=True)
splits = ["train", "test"]
subfolder = ["A", "B"]
for split in splits:
    for name in subfolder:
        os.makedirs(os.path.join(target, split, name), exist_ok=True)

remove_files = []      

moca_files = [os.path.join(moca_dir, f) for f in os.listdir(moca_dir) if f.lower().endswith(".png")]
static_files = [os.path.join(fu3_dir, f) for f in os.listdir(fu3_dir)]
static_files += [os.path.join(fu4_dir, f) 
             for f in os.listdir(fu4_dir) 
             if f not in remove_files]

random.shuffle(static_files)
random.shuffle(moca_files)

# split index
moca_split_idx = int(len(moca_files) * 0.2)
static_split_idx = int(len(static_files) * 0.2)

# split in train and test
train_moca = moca_files[:moca_split_idx]
test_moca = moca_files[moca_split_idx:]

train_static = static_files[:static_split_idx]
test_static = static_files[static_split_idx:]


# copy digital drawings in B
for src in train_moca:
    dst = os.path.join(target, "train", "B", os.path.basename(src))
    shutil.copy2(src, dst)

for src in test_moca:
    dst = os.path.join(target, "test", "B", os.path.basename(src))
    shutil.copy2(src, dst)

# copy scanned drawings in A
for src in train_static:
    dst = os.path.join(target, "train", "A", os.path.basename(src))
    shutil.copy2(src, dst)

for src in test_static:
    dst = os.path.join(target, "test", "A", os.path.basename(src))
    shutil.copy2(src, dst)

def check_overlap(train_dir, test_dir):
    train_files = set(os.listdir(train_dir))
    test_files = set(os.listdir(test_dir))
    
    overlap = train_files & test_files 
    
    if overlap:
        print(f"overlap detected ({len(overlap)} files) between {train_dir} and {test_dir}:")
        for f in overlap:
            print(f)
        return False
    else:
        print(f"no overlaps between {train_dir} and {test_dir}")
        return True


check_overlap(os.path.join(target, "train", "A"), os.path.join(target, "test", "A"))
check_overlap(os.path.join(target, "train", "B"), os.path.join(target, "test", "B"))





    