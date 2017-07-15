import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score


def f2_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def main():
    label = []
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
    for x in df_train_labels[:, 1:]:
        tmp = np.zeros(17)
        for y in range(17):
            tmp[y] = x[y]
        label.append(tmp)
    best = 0
    best_score = -1
    totry = np.arange(0, 1, 0.005)
    for t in totry:
        pred = []
        for x in df_train_predictions[:, 1:]:
            tmp = np.zeros(17)
            for y in range(17):
                if x[y] > t:
                    tmp[y] = 1.0
                else:
                    tmp[y] = 0.0
            pred.append(tmp)
        score = f2_score(label, pred)
        if score > best_score:
            best_score = score
            best = t
    print('Best score: ', best_score, ', Threshold: ', best)


if __name__ == '__main__':
    main()
