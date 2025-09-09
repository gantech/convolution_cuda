import matplotlib.pyplot as plt 
import pandas as pd

pdata = pd.read_csv('performance.txt')

fig = plt.figure()
plt.bar(range(len(pdata)), pdata.iloc[:, 3])
plt.xticks(range(len(pdata)), pdata.iloc[:, 1], rotation=45, ha='right', wrap=True)
plt.ylabel('GFLOPs')
plt.grid()
plt.tight_layout()
plt.savefig('flops.png')
plt.close(fig)

fig = plt.figure()
plt.bar(range(len(pdata)), pdata.iloc[:, 4])
plt.xticks(range(len(pdata)), pdata.iloc[:, 1], rotation=45, ha='right', wrap=True)
plt.ylabel('GB/s')
plt.grid()
plt.tight_layout()
plt.savefig('bandwidth.png')
plt.close(fig)

