import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    xvalue = list()
    yvalue = list()
    xarr = list()
    yarr = list()
    nxarr = list()
    nyarr = list()
    df_train_labels = pd.read_csv('train_labels.csv', header=None, dtype=object)
    df_train_predictions = pd.read_csv('labels_train_output.csv', header=None, dtype=object)
    df_train_labels = df_train_labels.as_matrix()
    df_train_predictions = df_train_predictions.as_matrix()
    for x in range(len(df_train_labels)):
        for y in range(1, len(df_train_labels[x])):
            df_train_labels[x][y] = float(df_train_labels[x][y])
    for x in range(len(df_train_predictions)):
        for y in range(1, len(df_train_predictions[x])):
            df_train_predictions[x][y] = float(df_train_predictions[x][y])
    for x in range(len(df_train_predictions[:, 1])):
        xvalue.append(df_train_labels[:, 17][x])
        yvalue.append(df_train_predictions[:, 17][x])

    # for x in range(len(xvalue)):
    #     if xvalue[x] == 0.0:
    #         print(yvalue[x])

    nparr = np.arange(0.001, 0.9, 0.001)
    for i in nparr:
        count = 0
        totalcount = 0
        for x in range(len(xvalue)):
            if xvalue[x] == 1.0:
                totalcount += 1
                if yvalue[x] > i:
                    count += 1
        print(count, totalcount)
        xarr.append(count)
        yarr.append(totalcount)
        count = 0
        totalcount = 0
        for x in range(len(xvalue)):
            if xvalue[x] == 0.0:
                totalcount += 1
                if yvalue[x] > 0.073:
                    count += 1
        # print(count, totalcount)
        nxarr.append(count)
        nyarr.append(totalcount)
    plt.plot(xarr, yarr)
    plt.show()


if __name__ == '__main__':
    main()
