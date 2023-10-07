import os
import shutil
from tqdm import tqdm

def main():
    path = '/home/vankhoa/datasets/CrowdHuman/crowdhuman-608x608'
    output_dir = '/home/vankhoa/datasets/CrowdHuman/crowdhuman'

    # read train.txt
    with open(path + '/train.txt', 'r') as f:
        train = f.readlines()
    train = [x.strip() for x in train]

    train_split = [x.split('/')[-1] for x in train]
    print(train_split[:10])

    # read val.txt
    with open(path + '/test.txt', 'r') as f:
        val = f.readlines()
    val = [x.strip() for x in val]

    val_split = [x.split('/')[-1] for x in val]
    print(val_split[:10])

    os.makedirs(output_dir + '/train', exist_ok=True)
    os.makedirs(output_dir + '/valid', exist_ok=True)
    
    train_dir = output_dir + '/train'
    valid_dir = output_dir + '/valid'
    train_img_dir = train_dir + '/images'
    train_lb_dir = train_dir + '/labels'

    valid_img_dir = valid_dir + '/images'
    valid_lb_dir = valid_dir + '/labels'

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lb_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lb_dir, exist_ok=True)

    for file in tqdm(train_split):
        shutil.copy(path + '/' + file, train_img_dir)
        shutil.copy(path + '/' + file.replace('jpg','txt'), train_lb_dir)

    for file in tqdm(val_split):
        shutil.copy(path + '/' + file, valid_img_dir)
        shutil.copy(path + '/' + file.replace('jpg','txt'), valid_lb_dir)

if __name__ == '__main__':
    main()