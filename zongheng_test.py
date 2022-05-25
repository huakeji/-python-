import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math

pd_columns = pd.read_excel(
    r'C:\Users\Administrator\Desktop\zh_data\评价指标_2.xlsx')
# print(pd_columns.groupby(['目标层', '准则层', '一级指标', '二级指标']).sum())
# pd.read_excel(r'C:\Users\Administrator\Desktop\zh_data\评价指标体系分省2011-2018.xlsx', sheet_name='北京')
time_columns = ['2010', '2011',  '2012',   '2013',   '2014',   '2015', '2016',  '2017', '2018', '2019']
city_columns = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西',
                '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']

list_zongheng = []
for i in time_columns:
    temp_pd = pd.read_excel(
        r'C:\Users\Administrator\Desktop\zh_data\20220513-example.xlsx', sheet_name=i)
    for j in temp_pd.columns:
        if j == '省市' or j == '时间':
            continue
        temp_pd = temp_pd.fillna(np.mean(temp_pd[j]))
    list_zongheng.append(temp_pd)
list_zongheng111 = []  # 归一化后每一年数据，有7年数据。
for t in range(len(list_zongheng)):
    temp_pd = list_zongheng[t]
    temp_pd_1 = temp_pd.copy()
    xmin = temp_pd_1.min()
    xmax = temp_pd_1.max()
    for i in temp_pd['省市'].values:
        for j in temp_pd.columns:
            if '省市' == j or '时间' == j:
                continue
            elif '城乡居民人均可支配收入之比' == j or '城乡居民人均消费支出之比' == j\
                    or '化肥施用强度' == j or '农药使用强度' == j or '农膜使用效率' == j:
                temp_pd_1.loc[temp_pd_1['省市'] == i, j] = 1 + 2 * \
                     (xmax[j] - temp_pd.loc[temp_pd['省市'] == i, j].values[0]
                      ) / (xmax[j] - xmin[j])
            else:
                temp_pd_1.loc[temp_pd_1['省市'] == i, j] = 1 + 2 * \
                    (temp_pd.loc[temp_pd['省市'] == i, j].values[0] -
                     xmin[j])/(xmax[j]-xmin[j])
    list_zongheng111.append(temp_pd_1)

# for m in range(len(list_zongheng111)):
#     data = list_zongheng111[m]
#     pd_data = pd.DataFrame(data)
#     pd_data.to_excel(r'C:\Users\Administrator\Desktop\pd_data'+str(m)+'.xlsx')
print('共处理', '、'.join([str(i) for i in time_columns]).strip(
    '、'), '等', len([str(i) for i in time_columns]), '个时间维度数据。')


def con(w):  # 约束条件
    cons = ([{'type': 'eq', 'fun': lambda w: np.dot(np.mat(w), np.mat(w).T).squeeze().sum()-1},  # 等式约束
             {'type': 'ineq', 'fun': lambda w: w}])  # 不等式约束
    return cons


def YHQU2(w):  # 二级指标
    sum_x = 0
    for i in range(len(list_zongheng111)):
        x = list_zongheng111[i]
        x_mat = np.mat(x.loc[:, columns_x])
        sum_x += (np.mat(w)*np.dot(x_mat.T, x_mat)*np.mat(w).T).squeeze().sum()
    return -sum_x


def YHQU1(w):  # 一级指标
    sum_x = 0
    for i in time_columns:
        x = pd.read_excel(r'C:\Users\Administrator\Desktop\zh_data\一级指标/'+j+str(i)+'.xlsx')
        x_mat = np.mat(x.loc[:, columns_x])
        sum_x += (np.mat(w)*np.dot(x_mat.T, x_mat)*np.mat(w).T).squeeze().sum()
    return -sum_x


def YHQUZHUNZE(w):  # 准则层
    sum_x = 0
    for i in time_columns:
        x = pd.read_excel(r'C:\Users\Administrator\Desktop\zh_data\准则层\目标层'+str(i)+'.xlsx')
        x_mat = np.mat(x.loc[:, columns_x])
        sum_x += (np.mat(w)*np.dot(x_mat.T, x_mat)*np.mat(w).T).squeeze().sum()
    return -sum_x


list_zongheng_zhunze = {}
pd_mubiao = pd.DataFrame(columns=pd_columns['目标层'].unique())
dict_z_value = {}
for ij in range(len(list_zongheng111)):
    list_zongheng_yiji = {}
    pd_zhunze = pd.DataFrame(columns=pd_columns['准则层'].unique())
    dict_1_value = {}
    dict_z = {}
    for j in pd_columns['准则层'].unique():
        dict_1 = {}
        pd_yiji = pd.DataFrame(
            columns=pd_columns.loc[pd_columns['准则层'] == j, '一级指标'].unique())
        for x in pd_columns.loc[pd_columns['准则层'] == j, '一级指标'].unique():
            dict_2 = {}
            columns_x = pd_columns.loc[pd_columns['一级指标']
                                       == x, '二级指标'].unique()
            w0 = [1/len(columns_x)]*len(columns_x)
            cons = con(w0)
            temp_pd = list_zongheng111[ij]
            res = minimize(fun=YHQU2, x0=w0, method='SLSQP', constraints=cons)
            w = (res.x/sum(res.x))
            temp_pd_1 = temp_pd.loc[:, columns_x]*w
            temp_pd_1['时间'] = temp_pd['时间']
            temp_pd_1['省市'] = temp_pd['省市']
            temp_pd_1.to_excel(r'C:\Users\Administrator\Desktop\zh_data/二级指标/'+x+str(temp_pd['时间'].values[0])+'.xlsx', encoding='gbk')
            temp_pd_1[x] = temp_pd_1.loc[:, columns_x].sum(axis=1)
            pd_yiji[x] = temp_pd_1[x]
            pd_yiji['时间'] = temp_pd['时间']
            pd_yiji['省市'] = temp_pd['省市']
            list_zongheng_yiji[j] = pd_yiji
            pd_yiji.to_excel(
                r'C:\Users\Administrator\Desktop\zh_data/一级指标/'+j+str(temp_pd['时间'].values[0])+'.xlsx', encoding='gbk')
            for item_2, value_2 in zip(columns_x, res.x/sum(res.x)):
                dict_2[item_2] = value_2
            dict_1[x] = dict_2  # 先把这下面的注释掉，求出结果后再运行一遍
        columns_x = pd_columns.loc[pd_columns['准则层'] == j, '一级指标'].unique()
        w0 = [1/len(columns_x)]*len(columns_x)
        temp_pd = list_zongheng_yiji[j]
        cons = con(w0)
        res = minimize(fun=YHQU1, x0=w0, method='SLSQP', constraints=cons)
        for item_1, value_1 in zip(columns_x, res.x/sum(res.x)):
            dict_1_value[item_1] = value_1
        dict_z[j] = dict_1
        temp_pd_1 = temp_pd.loc[:, columns_x]*(res.x/sum(res.x))
        temp_pd_1['时间'] = temp_pd['时间']
        temp_pd_1['省市'] = temp_pd['省市']
        temp_pd_1[j] = temp_pd_1.loc[:, columns_x].sum(axis=1)
        pd_zhunze[j] = temp_pd_1[j]
        pd_zhunze['时间'] = temp_pd_1['时间']
        pd_zhunze['省市'] = temp_pd_1['省市']
        list_zongheng_zhunze[ij] = pd_zhunze
        pd_zhunze.to_excel(
            r'C:\Users\Administrator\Desktop\zh_data/准则层/目标层'+str(temp_pd['时间'].values[0])+'.xlsx', encoding='gbk')
    columns_x = pd_columns['准则层'].unique()  # 先把这下面的注释掉，求出结果后再运行一遍
    w0 = [1/len(columns_x)]*len(columns_x)
    temp_pd = list_zongheng_zhunze[ij]
    cons = con(w0)
    res = minimize(fun=YHQUZHUNZE, x0=w0, method='SLSQP', constraints=cons)
    for item_z, value_z in zip(columns_x, res.x/sum(res.x)):
        dict_z_value[item_z] = value_z
    temp_pd_1 = temp_pd.loc[:, columns_x]*(res.x/sum(res.x))
    temp_pd_1['时间'] = temp_pd['时间']
    temp_pd_1['省市'] = temp_pd['省市']
    pd_mubiao['高质量发展'] = temp_pd_1.loc[:, columns_x].sum(axis=1)
    pd_mubiao['时间'] = temp_pd['时间']
    pd_mubiao['省市'] = temp_pd['省市']
    pd_mubiao.to_excel(
        r'C:\Users\Administrator\Desktop\zh_data/高质量发展/最终值'+str(temp_pd['时间'].values[0])+'.xlsx', encoding='gbk')
pd_result = pd.DataFrame(
    columns=['省市']+time_columns)
for i in time_columns:
    temp_pd = pd.read_excel(r'C:\Users\Administrator\Desktop\zh_data/高质量发展/最终值'+str(i)+'.xlsx')
    del temp_pd['Unnamed: 0']
    pd_result['省市'] = temp_pd['省市']
    pd_result.loc[:, i] = temp_pd['高质量发展']
pd_zhibiao_quanzhong = pd.DataFrame(
    columns=['目标层', '准则层', '准则层权重', '一级指标', '一级指标权重', '二级指标', '二级指标权重'])
for i, j in dict_z.items():
    for x, y in j.items():
        for s, t in y.items():
            ss = pd.Series({'目标层': '高质量发展', '准则层': i, '准则层权重': dict_z_value.get(
                i), '一级指标': x, '一级指标权重': dict_1_value.get(x), '二级指标': s, '二级指标权重': t})
            pd_zhibiao_quanzhong = pd_zhibiao_quanzhong.append(
                ss, ignore_index=True, sort=False)


pd_zhibiao_quanzhong.groupby(['目标层','准则层','准则层权重','一级指标','一级指标权重','二级指标','二级指标权重']).sum().to_excel(
    r'C:\Users\Administrator\Desktop\zh_data/指标权重.xlsx',encoding='gbk')


def fun_time(w):  # I值，时间尺度
    columns_x = time_columns
    sum_w = 0
    for i in range(len(columns_x)):
        sum_w += -w[i]*math.log(w[i])
    return -(sum_w)


def con3(w):  # lambda值
    sum_w = 0
    for k in range(1, len(time_columns)+1):
        sum_w += (len(time_columns)-k)/(len(time_columns)-1)*w[k-1]
    return sum_w


def con2(w):
    cons = ([{'type': 'eq', 'fun': lambda w: con3(w)-0.35},
             {'type': 'eq', 'fun': lambda w: sum(w)-1}])
    return cons


w0 = [1/len(time_columns)]*len(time_columns)
cons2 = con2(w0)
res = minimize(fun=fun_time, x0=w0, method='SLSQP', constraints=cons2)
wt = res.x/sum(res.x)


pd_time_w = pd.DataFrame(columns=time_columns, index=['时间权重'])
for i, j in zip(pd_time_w.columns, wt):
    pd_time_w.loc['时间权重', i] = j


pd_time_w.to_excel(r'C:\Users\Administrator\Desktop\zh_data\时间权重.xlsx', encoding='gbk')


columns_x = list(time_columns)
r1 = (((wt*pd_result.loc[:, columns_x]) -
       (wt*pd_result.loc[:, columns_x]).mean())**2).sum(axis=1)
r2 = (((pd_result.loc[:, columns_x]**wt) -
       (pd_result.loc[:, columns_x]**wt).mean())**2).sum(axis=1)
a1 = r1/(r1+r2)
a2 = r2/(r1+r2)


pd_result['最终测度值'] = a1*(wt*pd_result.loc[:, columns_x]).sum(axis=1) + \
    a2*(pd_result.loc[:, columns_x]**wt).prod(axis=1)
# pd_result['最终测度值'] = (wt*pd_result.loc[:, columns_x]).sum(axis=1)
pd_result['排名'] = pd_result['最终测度值'].rank(method='first', ascending=False)
pd_result.to_excel(r'C:\Users\Administrator\Desktop\zh_data\最终值.xlsx', encoding='gbk')

pd_result.loc[pd_result['省市'].isin(['北京', '天津', '河北', '山西', '内蒙古']), '区域'] = '华北地区'
pd_result.loc[pd_result['省市'].isin(['辽宁', '吉林', '黑龙江']), '区域'] = '东北地区'
pd_result.loc[pd_result['省市'].isin(['河南', '湖北', '湖南']), '区域'] = '华中地区'
pd_result.loc[pd_result['省市'].isin(['广东', '广西', '海南']), '区域'] = '华南地区'
pd_result.loc[pd_result['省市'].isin(['重庆', '四川', '贵州', '云南', '西藏']), '区域'] = '西南地区'
pd_result.loc[pd_result['省市'].isin(['陕西', '甘肃', '青海', '宁夏', '新疆']), '区域'] = '西北地区'
pd_result.loc[pd_result['省市'].isin(['上海', '江苏', '浙江', '安徽', '福建', '江西', '山东']), '区域'] = '华东地区'
result = pd_result.groupby(['区域', '省市']).mean()
result.to_excel(r'C:\Users\Administrator\Desktop\zh_data\result.xlsx', encoding='gbk')





