import os
import cv2
import h5py
import pickle
import pandas as pd


def main():
    label_dict = {}
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    filecount = 0
    evalcount = 0
    labels = {'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5,
              'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11,
              'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
    df = pd.read_csv('train_v2.csv', header=None, dtype=object)
    for x in df.as_matrix()[1:]:
        # print(x[0])
        label_lst = []
        for y in str(x[1:][0]).split(' '):
            label_lst.append(labels[y])
        # print(label_lst)
        label_dict[x[0]] = label_lst
    # print(label_dict)
    explore_path = "/Users/praneet/Documents/Kaggle/Amazon/train"
    for root, dirs, files in os.walk(explore_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                filecount += 1
                print(file_name)
                file_path = os.path.abspath(os.path.join(root, file_name))
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fixed_size = (224, 224)
                img = cv2.resize(img, dsize=fixed_size)
                # print(img.shape)
                # print(img)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                file_title = file_name.split('.')[0]
                # print(label_dict[file_title])
                train_data.append(img)
                train_label.append(label_dict[file_title])
                if filecount % 11 == 0:
                    evalcount += 1
                    eval_data.append(img)
                    eval_label.append(label_dict[file_title])
    file_train_data = []
    file_train_label = []
    file_eval_data = []
    file_eval_label = []
    print(filecount)
    for x in range(len(train_label)):
        for y in range(len(train_label[x])):
            file_train_data.append(train_data[x])
            file_train_label.append(train_label[x][y])
    print(len(file_train_data))
    print(len(file_train_label))
    file = h5py.File('train_data.hdf5', 'a')
    file.create_dataset('train', data=file_train_data)
    file.close()
    with open('train_label', 'ab') as fp:
        pickle.dump(file_train_label, fp)
    print(evalcount)
    for x in range(len(eval_label)):
        for y in range(len(eval_label[x])):
            file_eval_data.append(eval_data[x])
            file_eval_label.append(eval_label[x][y])
    print(len(file_eval_data))
    print(len(file_eval_label))
    file = h5py.File('eval_data.hdf5', 'a')
    file.create_dataset('eval', data=file_eval_data)
    file.close()
    with open('eval_label', 'ab') as fp:
        pickle.dump(file_eval_label, fp)


if __name__ == '__main__':
    main()
