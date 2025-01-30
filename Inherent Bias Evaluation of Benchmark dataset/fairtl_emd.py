import numpy as np
from scipy.optimize import linprog
import pandas as pd


def calculate_p_distribution(df, attr, category, target=None, subgroup=None):

    if subgroup is not None:
        df = df[df[attr] == subgroup]

    if target is not None:
        attr = target


    # category = np.array(df[attr].unique())
    
    distr = {}
    
    for i in category:
        mask = (df[attr] == i).sum()
        distr[i] = mask/df.shape[0]

    return distr



def Manhattan_distance(distr1, distr2):
    dis = np.zeros((len(distr1), len(distr2)))

    for i in range(len(distr1)):
        for j in range(len(distr2)):
            dis[i][j] = abs(i-j)

    return dis


def calculate_EMD(distr, target):
    
    dis = Manhattan_distance(distr, target)

    C = dis.flatten()

    if isinstance(distr, dict):

        distr = list(distr.values())

    if isinstance(target, dict):
        target = list(target.values())

    m = len(distr)
    n = len(target)
    A_eq = np.zeros((m + n, m * n))
    b_eq = np.zeros(m + n)

    for i in range(m):
        A_eq[i, i * n: (i+1)*n] = 1
        b_eq[i] = distr[i]

    for j in range(n):
        A_eq[m + j, j::n] = 1
        b_eq[m + j] = target[j]

    bounds = [(0, None)] * (m * n)
    result = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return result.fun, result.x.reshape(m, n)
    else:
        print("Optimization failed:", result.message)


    
def EMD(df, attr, target, subgroup, permu_num=1000):

    # category = np.array(df[target].unique())

    category = purmutation_category(df, target)

    orgin = [9999]
    best_category = 0

    for item in category:

        distr_p = calculate_p_distribution(df, attr, item, target, subgroup)
        target_p = calculate_p_distribution(df, target, item)

        step_emd = calculate_EMD(distr_p, target_p)

        if step_emd[0] < orgin[0]:
            orgin = step_emd
            best_category = item

    category = best_category


    df_attr = df[df[attr] == subgroup]
    combined = pd.concat([df_attr, df], axis=0, ignore_index=True)
    
    permu = []
    subgroup_num = df_attr.shape[0]
    
    for i in range(permu_num):

        df_shuffled = combined.sample(frac=1, random_state=i).reset_index(drop=True)
 
        perm_group1 = df_shuffled.iloc[:subgroup_num]
        perm_group2 = df_shuffled.iloc[subgroup_num:]

        permu_distr = calculate_p_distribution(perm_group1, target, category)
        permu_target = calculate_p_distribution(perm_group2, target, category)

        permu_emd = calculate_EMD(permu_distr, permu_target)[0]
        permu.append(permu_emd)


    extreme_num = 0
    for i in permu:
        if i >= orgin[0]:
            extreme_num += 1
    
    p_value = extreme_num/ permu_num
    return orgin[0], p_value, orgin[1]



def statistical_dist(df, target, purmu_num = 1000, **kw):

    result = {'Subgroup': [], 'EMD': [], 'Count': [], 'P-Value': []}

    for key in kw:
        subgroup = kw[key]

        for i in subgroup:

            count = (df[key] == i).sum()
            emd, p_value, _ = EMD(df, key, target, i, purmu_num)

            result['Subgroup'].append(i)
            result['EMD'].append(emd)
            result['Count'].append(count)
            result['P-Value'].append(p_value)

    df = pd.DataFrame(result)
    return df


        
def get_category(df, target, descending=True):
    lst = list(df[target].unique())
    length = len(lst)

    if descending:
        order = {}
        for i in lst:
            count = (df[target] == i).sum()
            order[i] = count

        lst = []
        for i in range(length):
            max_key = max(order, key=order.get)
            lst.append(max_key)
            del order[max_key] 

    return lst


def purmutation_category(df, target):
    lst = list(df[target].unique())

    results = []

    def trace(path, choice):

        if len(path) == len(lst):
            results.append(list(path))
            return

        for i in choice:
            if i in path:
                continue
            path.append(i)
            trace(path, choice)
            path.pop()
    trace([], lst)
    return results




