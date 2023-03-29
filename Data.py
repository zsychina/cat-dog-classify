import os
import shutil

def movefile(srcpath, dstpath):
    if not os.path.isfile(srcpath):
        raise('file not exists')
    else:
        fpath, fname = os.path.split(dstpath)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcpath, dstpath)

root = './data/train/'

train_files = os.listdir(root + 'train/')


for train_file in train_files:
    srcpath = os.path.join(root, 'train', train_file)
    file = train_file.split('.')
    if file[0] == 'cat':
        dstpath = os.path.join(root, 'cat', train_file)
    elif file[0] == 'dog':
        dstpath = os.path.join(root, 'dog', train_file)
    else:
        raise ValueError('Invalid file name')
    
    movefile(srcpath, dstpath)

