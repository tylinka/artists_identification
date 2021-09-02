import os
import shutil
import numpy as np


root_dir = '/content/drive/MyDrive/education/images/images/'  # data root path
classes_dir = [name for name in os.listdir(".")]  # total labels

val_ratio = 0.15
test_ratio = 0.05

for cls in classes_dir:
    os.makedirs(root_dir + 'train/' + cls)
    os.makedirs(root_dir + 'val/' + cls)
    os.makedirs(root_dir + 'test/' + cls)

# Creating partitions of the data after shuffeling
for cls in classes_dir:
    src = root_dir + cls  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + 'train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir + 'val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + 'test/' + cls)
