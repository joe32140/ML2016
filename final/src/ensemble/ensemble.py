import numpy as np

f1=open('./1', 'r')
f2=open('./2', 'r')
f3=open('./3', 'r')

f1.readline()
f2.readline()
f3.readline()

ans1 = []
for line in f1:
    ans1.append(int(line.strip().split(',')[1]))

ans2 = []
for line in f2:
    ans2.append(int(line.strip().split(',')[1]))

ans3 = []
for line in f3:
    ans3.append(int(line.strip().split(',')[1]))

ans = []
for x in range(len(ans1)):
    tmp=np.zeros(5)
    tmp[ans1[x]] += 1
    tmp[ans2[x]] += 1
    tmp[ans3[x]] += 1
    ans.append(np.argmax(tmp))

with open('ans', 'w') as out:
    out.write('id,label\n')
    cnt=1
    for x in ans:
        out.write(str(cnt)+','+str(x)+'\n')
        cnt+=1
