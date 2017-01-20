import numpy as np
import cPickle, pickle
from keras.utils import np_utils

with open("train", 'r') as f:
    txt = f.readlines()

with open("training_attack_types.txt", "r") as f:
    atk_dic = {}
    for line in f.readlines():
        line = line.strip().split()
        if line[1] in atk_dic.keys():
            if line[0] not in atk_dic[line[1]]:
                atk_dic[line[1]].append(line[0])
        else:
            atk_dic[line[1]]=[]
            atk_dic[line[1]].append(line[0])
#print atk_dic
col1 = ['udp', 'icmp', 'tcp']
col2 = ['aol', 'urp_i', 'netbios_ssn', 'Z39_50', 'smtp', 'domain', 'private', 'echo', 'printer', 'red_i', 'eco_i', 'ftp_data', 'sunrpc', 'urh_i', 'uucp', 'pop_3', 'pop_2', 'systat', 'ftp', 'sql_net', 'whois', 'tftp_u', 'netbios_dgm', 'efs', 'remote_job', 'daytime', 'pm_dump', 'other', 'finger', 'ldap', 'netbios_ns', 'kshell', 'iso_tsap', 'ecr_i', 'nntp', 'http_2784', 'shell', 'domain_u', 'uucp_path', 'courier', 'exec', 'tim_i', 'netstat', 'telnet', 'gopher', 'rje', 'hostnames', 'link', 'ssh', 'http_443', 'csnet_ns', 'X11', 'IRC', 'harvest', 'login', 'supdup', 'name', 'nnsp', 'mtp', 'http', 'ntp_u', 'bgp', 'ctf', 'klogin', 'vmnet', 'time', 'discard', 'imap4', 'auth', 'http_8001']
col3 = ['OTH', 'RSTR', 'S3', 'S2', 'S1', 'S0', 'RSTOS0', 'REJ', 'SH', 'RSTO', 'SF'] 

text = []
for line in txt:
    line = line.strip(".\n").split(",")
    
    line[1] = col1.index(line[1])
    line[2] = col2.index(line[2])
    line[3] = col3.index(line[3])
    if line[-1] in atk_dic["dos"]:
        line[-1] = 1
    elif line[-1] in atk_dic["u2r"]:
        line[-1] = 2
    elif line[-1] in atk_dic["r2l"]:
        line[-1] = 3
    elif line[-1] in atk_dic["probe"]:
        line[-1] = 4
    else:
        line[-1] = 0

    text.append(line)

print text[1]
text = np.array(text)

text = np.concatenate((text, np_utils.to_categorical(text[:, 1], 3)), axis=1)
text = np.concatenate((text, np_utils.to_categorical(text[:, 2], 70)), axis=1)
text = np.concatenate((text, np_utils.to_categorical(text[:, 3], 11)), axis=1)
text = np.delete(text, [1, 2, 3], 1)
text = text.astype(np.float) 
print text.shape

np.save('train_sparse.npz', text)

#f = open('train_sparse.p', 'wb')
#pickle.dump(text, f)
#f.close()
#col 1 has 3 unique value ['udp', 'icmp', 'tcp']
#col 2 has 9 unique value ['ftp', 'domain_u', 'smtp', 'http', 'private', 'other', 'eco_i', 'ftp_data', 'ecr_i']
#col 3 has 3 unique value ['REJ', 'S0', 'SF']


