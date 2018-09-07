import os
import pandas as pd

fpath1 = input("Enter file path: ")
if len(fpath1) < 1 : fpath1 = "D:\\HYHY\\数据\\20180316\\第一批\\"
fpath2 = input("Enter file path: ")
if len(fpath2) < 1 : fpath2= "D:\\HYHY\\数据\\20180316\\第二批\\"

fs1 = os.listdir(fpath1); fs2 = os.listdir(fpath2)
data1 = pd.DataFrame(0, index = fs1+fs2, columns = ['CT/CTZ','CTC']);
data1.index = data1.index.rename('ID');data1.index = data1.index.astype('int64')

for j in [fpath1, fpath2] :
    fs = os.listdir(j)
    for i in fs :
        tmp_path = os.path.join(j,i)
        i = int(i)
        if 'CT' in os.listdir(tmp_path) : data1.loc[i,:]['CT/CTZ'] = 1
        if 'CTZ' in os.listdir(tmp_path) : data1.loc[i,:]['CT/CTZ'] = 1
        if 'CTC' in os.listdir(tmp_path) : data1.loc[i,:]['CTC'] = 1

data2 = pd.read_excel('D:\\HYHY\\数据\\20180316\\info.xls','Sheet1')

id1 = sorted(list(data1.index))
id2 = sorted(list(data2.iloc[:,0].drop_duplicates()))

InfNix = list(); InxNif = list()
for i in id1 :
    if not i in id2 : InfNix.append(i)
for i in id2 :
    if not i in id1 : InxNif.append(i)

if not len(InfNix) is 0 : pd.DataFrame(InfNix, columns = ['ID']).to_csv("D:\\HYHY\\数据\\20180316\\InfileNotxls.csv")
if not len(InxNif) is 0 : pd.DataFrame(InxNif, columns = ['ID']).to_csv("D:\\HYHY\\数据\\20180316\\InxlsNotfile.csv")
