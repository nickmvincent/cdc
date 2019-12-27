#%%
import glob
import pandas as pd
import os

#%%
#labels = list(range(0,4))
labels = [2, 3]
print(labels)
new_labels = {}
for i, label in enumerate(labels):
    new_labels[label] = i

print(new_labels)

#%%
newdir = 'custom'
names = pd.read_csv('./coco.names', header=None)
names.iloc[labels].to_csv(f'{newdir}/classes.names', header=None, index=None)


#%%
configs = [
    ('coco/labels/train2014', f'{newdir}/train.txt'),
    ('coco/labels/val2014', f'{newdir}/valid.txt'),
]



for folder, txt_file in configs:
    kept = []
    n_ditched = 0

    new_folder = folder.replace('coco', newdir)
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    new_image_folder = new_folder.replace('labels', 'images')
    if not os.path.isdir(new_image_folder):
        os.mkdir(new_image_folder)

    

    for path in glob.glob(folder + '/*'):
        old_img_file = path.replace('labels', 'images').replace('.txt', '.jpg')
        new_label_file = path.replace('coco', newdir)
        new_img_file = old_img_file.replace('coco', newdir)
        with open(old_img_file, 'rb') as r:
            with open(new_img_file, 'wb') as w:
                w.write(r.read())
        # probably slow, but fast to code. Change if needed.
        df = pd.read_csv(path, header=None, sep=' ')
        
        # we can throw away labels for classes we don't care about.
        lines_to_keep = df[0].isin(labels)

        if any(lines_to_keep):
            kept.append(f'data/{new_img_file}')
            df = df[lines_to_keep]
            #print(df[0])
            df[0] = df[0].map(new_labels).astype(int)
            df.to_csv(new_label_file, index=None, header=None, sep=' ')
        else:
            n_ditched += 1
        print(len(kept), n_ditched)
        pd.Series(kept).to_csv(txt_file, index=None)
        if len(kept) > 50:
            break

# %%
