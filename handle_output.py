import pandas as pd


def main():
    labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
              'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
              'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    # threshold = [0.013, 0.03, 0.03, 0.0305, 0.045, 0.0185, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    #              0.03, 0.03, 0.03]
    threshold = [0.03] * 17
    output_list = list()
    output_list.append(['image_name', 'tags'])
    df = pd.read_csv('output_actual.csv', header=None, dtype=object)
    df = df.as_matrix()
    for x in range(len(df)):
        for y in range(1, len(df[x])):
            df[x][y] = float(df[x][y])
    for x in range(df.shape[0]):
        tmp = []
        for y in range(1, 18):
            if df[x][y] > threshold[y - 1]:
                tmp.append(1)
            else:
                tmp.append(0)
        opstr = ''
        for z in range(len(tmp)):
            if tmp[z] == 1:
                opstr = opstr + ' ' + labels[z]
        opstr = opstr.strip()
        newtmp = list()
        val = df[x][0].split('.')[0]
        newtmp.append(val)
        newtmp.append(opstr)
        print(newtmp)
        output_list.append(newtmp)
    newdf = pd.DataFrame(output_list)
    newdf.to_csv('submit_file.csv', index=False, header=False)


if __name__ == '__main__':
    main()
