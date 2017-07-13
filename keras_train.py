import os
import cv2
import time
import numpy as np
import pandas as pd
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization


def main():
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    label_dict = {}
    train_data = []
    train_label = []
    train_label_values = []
    filecount = 0
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
                filecount += 1
                print(file_name)
                file_path = os.path.abspath(os.path.join(root, file_name))
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fixed_size = (64, 64)
                img = cv2.resize(img, dsize=fixed_size)
                file_title = file_name.split('.')[0]
                train_data.append(img)
                train_label.append(label_dict[file_title])
    train_data = np.array(train_data, dtype=np.float32)
    for x in train_label:
        lst = [0] * 17
        for y in x:
            lst[y] = 1
        train_label_values.append(lst)
    epochs_arr = [20, 5, 5]
    learn_rates = [0.001, 0.0001, 0.00001]
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        opt = optimizers.Adam(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=2),
                     ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=2)]
        model.fit(x=train_data, y=train_label_values, batch_size=128, verbose=2, epochs=epochs,
                  callbacks=callbacks, shuffle=True)
    model.save('mymodel.h5')
    del model
    backend.clear_session()
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)


if __name__ == '__main__':
    main()
