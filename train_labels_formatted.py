import os
import numpy as np
import pandas as pd


def main():
    label_dict = {}
    filecount = 0
    train_insert = []
    labels = {'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5,
              'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11,
              'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
    df = pd.read_csv('train_v2.csv', header=None, dtype=object)
    for x in df.as_matrix()[1:]:
        label_lst = []
        for y in str(x[1:][0]).split(' '):
            label_lst.append(labels[y])
        label_dict[x[0]] = label_lst
    explore_path = "/Users/praneet/Documents/Kaggle/Amazon/train"
    for root, dirs, files in os.walk(explore_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                file_train_label = []
                tmp = []
                insert_labels = np.zeros(17)
                filecount += 1
                print(file_name)
                file_title = file_name.split('.')[0]
                file_train_label.append(label_dict[file_title])
                for x in file_train_label[0]:
                    insert_labels[x] = 1
                # print(file_name, insert_labels, file_train_label[0])
                tmp.append(file_name)
                for x in insert_labels:
                    tmp.append(x)
                train_insert.append(tmp)
    print(filecount)
    df = pd.DataFrame(train_insert)
    df.to_csv('train_predictions.csv', index=False, header=False)


if __name__ == '__main__':
    main()
