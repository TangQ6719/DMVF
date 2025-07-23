#先读取表格的数据
#我们定义1为正样本疾病样本，0为负样本阴性样本
import pandas as pd
import numpy as np
#先读取表格的数据,顺便拿到表头
csv_g = pd.read_csv("mimic/output-ground-1.csv") # 真实报告结果路径
header = csv_g.columns.tolist()
print(header[1])
csv_r = pd.read_csv("mimic/precdition.csv") # 预测结果路径
print(len(csv_r))
ex = Exception('表格长度不一致')
if len(csv_g) != len(csv_r):
    raise ex
else:
    print("表格长度一致，可以开始")
# 两个表格内的数据一致，都是对应上的，其中空白就是NaN，1就是1.0，0就是0，0
# 我们现在要做的是对比两个表格里面的数据，首先要确认两个表格都有的数据
# 我们先拿到两个表格中每一行的数据
#for i in range(0,len(csv1)):

#------------------------------------------------------这里是宏观指标
macro_precision = 0 ; macro_recall = 0 ; macro_f1 = 0; AUCs = []
for n in header:
    if n == "Reports" or n == "No Finding":
        continue
    # 拿到每一列的文件
    print("遍历列表为" + n)
    data_g_n = np.array(csv_g[n])
    data_r_n = np.array(csv_r[n])
    # print(data_g_n[0]) # 1.0可以用1代替
    # print(data_r_n[0]) # nan用！=1和0代替，将-1页包含进去
    # print(data_g_n[0]==data_r_n[0]) # 判断空白的方法
    print("-------------------------------------------------")
# 我们现在拿到两列数据先以这两类来计算指标，在把所有的类直接循环一下就可以得出宏观结果
# 我们要拿到等于1或者等于0两种结果，并且两个都有才行，其他的结果都扔掉
# 定义数组，宏观需要每一列都重新定义
    TP = 0 ; TN = 0 ; FP = 0 ; FN = 0


    # 以下表达式可以帮我们筛选出可以用的临床指标，我们把就可以用的来判断结果，然后将所有的结果先算出p、r和f1
    # 前提是每一列会有不同的情况，我们要分开设计
    for i in range(1,len(data_g_n)):
        # if (data_r_n[i]==1 or data_r_n[i]==0) and (data_g_n[i]==1 or data_g_n[i]==0):
        if data_g_n[i] == data_r_n[i]: # 不管1还是0，只要对了就是tp
            TP = TP + 1
        elif (data_g_n[i] == 1 or data_g_n[i] ==0) and np.isnan(data_r_n[i]): # 漏诊应该是真实报告检测出病情，但是生成报告没有，错误预测成负样本
            FN = FN + 1
        elif np.isnan(data_g_n[i]) and np.isnan(data_r_n[i]): #这个对最后的结果没影响，写出来备用
            TN = TN + 1
        elif (np.isnan(data_g_n[i]) and data_r_n[i] == 1 ) or (data_g_n[i] == 1 and data_r_n[i] == 0 ) \
                or (data_g_n[i] == 0 and data_r_n[i] == 1 ): # 属于错诊，原本没有或者不一致
            FP = FP + 1
    if TP ==0 :
        continue
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    #FPr = FP/(TN+FP)
    f1 = 2*(precision*recall)/(precision+recall)
    #print(recall,FPr)
    #AUC = np.trapz(recall, FPr)
    print(precision) # 这里是一类的召回率，我们直接算一下全部召回率
    print(recall)
    #print(AUC)
    macro_precision = macro_precision + precision
    macro_recall = macro_recall + recall
    macro_f1 = macro_f1 + f1
    #AUCs.append(AUC)
    print(1)
    print(TP)
print("宏观精确率：")
print(macro_precision/(len(header)-2))
print("宏观召回率：")
print(macro_recall/(len(header)-2))
print("宏观f1：")
print(macro_f1/(len(header)-2))
#print("AUC:")
#print(np.mean(AUCs))

# 把每一类的计算在将所有类进行整合，也就是先视为二分类



