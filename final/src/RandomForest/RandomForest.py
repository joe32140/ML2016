import os
import pickle
import sys

import numpy as np
import xgboost as xgb

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

x_train = []
y_train = []
y2_train = []
x_test = []
y_test = []
y2_test = []
model = []
protocol_id = {}
service_id = {}
flag_id = {}
attack_id = {}
type_map = {}
label_id = {
        'normal': 0,
        'dos': 1,
        'u2r': 2,
        'r2l': 3,
        'probe': 4
        }


def read_type_data(path):
    print('read type data from ' + path)

    global type_map, attack_id, label_id, protocol_id, service_id, flag_id

    type_file = 'training_attack_types.txt'
    protocol_file = 'protocol_id'
    service_file = 'service_id'
    flag_file = 'flag_id'

    with open(os.path.join(path, type_file), 'r') as file:
        for i, line in enumerate(file):
            split = line.strip().split()
            attack_id[split[0]] = i + 1
            type_map[i + 1] = label_id[split[1]]
        attack_id['normal'] = 0
        type_map[0] = 0

    with open(os.path.join(path, protocol_file), 'r') as file:
        for line in file:
            line = line.strip().split()
            protocol_id[line[0]] = int(line[1])

    with open(os.path.join(path, service_file), 'r') as file:
        for line in file:
            line = line.strip().split()
            service_id[line[0]] = int(line[1])

    with open(os.path.join(path, flag_file), 'r') as file:
        for line in file:
            line = line.strip().split()
            flag_id[line[0]] = int(line[1])


def read_train_data(path):
    print('read train data from ' + path)

    global x_train, y_train, y2_train, attack_id, protocol_id, service_id, flag_id

    data_name = 'train'
    
    with open(os.path.join(path, data_name), 'r') as file:
        for i, line in enumerate(file):
            data = line.strip()[:-1].split(',')
            x = [data[0]]

            # protocol type to one-hot
            tmp_list = [0] * len(protocol_id)
            tmp_list[protocol_id[data[1]]] = 1
            x.extend(tmp_list)

            # service type to one-hot
            tmp_list = [0] * len(service_id)
            tmp_list[service_id[data[2]]] = 1
            x.extend(tmp_list)

            # flag type to one-hot
            tmp_list = [0] * len(flag_id)
            tmp_list[flag_id[data[3]]] = 1
            x.extend(tmp_list)

            x.extend(data[4:-1])
            x_train.append(x)
            y_train.append(type_map[attack_id[data[-1]]])
            
            if i % 500000 == 0:
                print(i)

    x_train = np.array(x_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)


def train():
    print('train')

    global x_train, y_train, y2_train, model

    # Random Forest
    model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                oob_score=True,
                n_jobs=-1,
                verbose=2
                )
    model.fit(x_train, y_train)

    #score = model.oob_score_
    #print(score)


def output_model(path):
    print("output " + path)

    global model

    pickle.dump(model, open(path, 'wb'))


def load_model(path):
    print("load " + path)
    
    global model

    model = pickle.load(open(path, 'rb'))


def read_test_data(path):
    print('read test data from ' + path)

    global x_test, protocol_id, service_id, flag_id

    data_name = 'test.in'

    with open(os.path.join(path, data_name), 'r') as file:
        for i, line in enumerate(file):
            data = line.strip().split(',')
            x = [data[0]]

            # protocol type to one-hot
            tmp_list = [0] * len(protocol_id)
            tmp_list[protocol_id[data[1]]] = 1
            x.extend(tmp_list)

            # service type to one-hot
            tmp_list = [0] * len(service_id)
            if data[2] in service_id:
                tmp_list[service_id[data[2]]] = 1
            else:
                tmp_list[service_id['other']] = 1
            x.extend(tmp_list)

            # flag type to one-hot
            tmp_list = [0] * len(flag_id)
            tmp_list[flag_id[data[3]]] = 1
            x.extend(tmp_list)

            x.extend(data[4:])
            x_test.append(x)
            
    x_test = np.array(x_test).astype(np.float32)


def output_prediction(path):
    print('output ' + path)
    
    global x_test, y_test, y2_test, model, model2

    y_test = model.predict(x_test)

    with open(path, 'w') as file:
        file.write('id,label\n')
        for i, label in enumerate(y_test):
            #file.write('{},{}\n'.format(i + 1, type_map[label]))
            file.write('{},{}\n'.format(i + 1, int(label)))


def main():
    if len(sys.argv) == 3:
        read_type_data(sys.argv[1])
        read_train_data(sys.argv[1])
        train()
        output_model(sys.argv[2])
    elif len(sys.argv) == 4:
        load_model(sys.argv[2])
        read_type_data(sys.argv[1])
        read_test_data(sys.argv[1])
        output_prediction(sys.argv[3])
    else:
        print("Error")

if __name__ == '__main__':
    main()
