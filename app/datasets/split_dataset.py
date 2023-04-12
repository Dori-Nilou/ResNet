import os
import glob
import random
import shutil
from PIL import Image

def split_WHU_RS19():
    test_split_ratio = 0.3
    desired_size = 512
    # 需要进入到app/datasets目录下运行该程序
    raw_path = './WHU-RS19'

    # print(os.path.join(raw_path, '*'))
    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    # print(dirs)

    print('Totally: {len}, classes: {dirs}'.format(len=len(dirs), dirs=dirs))

    for path in dirs:
        path = path.split('/')[-1]
        print(path)

        os.makedirs(f'WHU-RS19-train-7/{path}', exist_ok=True)
        os.makedirs(f'WHU-RS19-test-3/{path}', exist_ok=True)

        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        # print(len(files))

        random.shuffle(files)

        boundary = int(len(files) * test_split_ratio)
        # print(boundary)

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            # print(img.size[0])
            old_size = img.size
            ratio = float(desired_size / max(old_size))
            new_size = tuple([int(x * ratio) for x in old_size])
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size - new_size[0])//2,
                                    (desired_size - new_size[1])//2))
            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'WHU-RS19-test-3/{path}', file.split('/')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(f'WHU-RS19-train-7/{path}', file.split('/')[-1].split('.')[0] + '.jpg'))

    print('spliting dataset WHU-RS19: end')



def main():
    split_WHU_RS19()
    print('split dataset WHU-RS19: succsess')


if __name__ == "__main__":
    main()
