import numpy as np
import sys

t = np.loadtxt(sys.argv[1])
t = t.T
col = t[int(sys.argv[2])]
col = np.sort(col)
#print col

ans=''
f = open('ans1.txt', 'wb')
for i in col:
    ans = ans + str(i)
    ans = ans+','

ans = ans[0:len(ans)-1]
f.write(ans)
f.close()
