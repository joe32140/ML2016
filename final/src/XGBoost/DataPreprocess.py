import sys
import os

data_name = 'train'

type_name = ['protocol_id', 'service_id', 'flag_id']
type_id = [{}, {}, {}]

count = [0, 0, 0]
with open(os.path.join(sys.argv[1], data_name), 'r') as file:
    for num, line in enumerate(file):
        line = line.strip().split(',')
        for i in range(3):
            if not line[i + 1] in type_id[i]:
                type_id[i][line[i + 1]] = count[i]
                count[i] += 1
        if num % 500000 == 0:
            print(num)


# add icmp to service_id
#type_id[1]['icmp'] = len(type_id[1])


for i in range(3):
    with open(os.path.join(sys.argv[1], type_name[i]), 'w') as file:
        for key, value in type_id[i].items():
            file.write('{} {}\n'.format(key, value))
