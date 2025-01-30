from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from pingouin import welch_anova
from scipy.stats import kruskal
import pandas as pd
import numpy as np



def build_models(seed):
    models = {
        'SVM': SVC(random_state=seed),
        'LR': LogisticRegression(random_state=seed),
        'KNN': KNeighborsClassifier(),
        'RF': RandomForestClassifier(random_state=seed),
        'DT': DecisionTreeClassifier(random_state=seed),
        'ANN': MLPClassifier(random_state=seed)
    }
    return models



def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity, recall, or true positive rate
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity or true negative rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False positive rate
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False negative rate

    return TPR, TNR, FPR, FNR, TP, TN, FP, FN



def run_experiment(kf, models, X_data, y_data, group_label, results_list):
    for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        fold_results = {'Fold': fold + 1, 'Group': group_label}
        print(f'Processing fold {fold + 1} for group {group_label}')
        for name, model in models.items():
            print(f'Training and evaluating model: {name}')

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            TPR, TNR, FPR, FNR, TP, TN, FP, FN = calculate_metrics(y_test, y_pred)

            fold_results.update({
                f'{name}_TPR': TPR, f'{name}_TNR': TNR,
                f'{name}_FPR': FPR, f'{name}_FNR': FNR,
                f'{name}_TP': TP, f'{name}_TN': TN,
                f'{name}_FP': FP, f'{name}_FN': FN
            })
        results_list.append(pd.DataFrame([fold_results]))



def is_normality(d):
    
    if len(set(d)) == 1:
        return -1

    _, p_sw = shapiro(d)
    
    return p_sw


def is_homogeneity_variances(protected, unprotected):

    _, p_lv = levene(protected, unprotected)

    return p_lv


def perform_t_tests(df, algorithm, label):

    # prevent division by zero error
    eps = 1e-10

    tpr_col = f"{algorithm}_TPR"
    fpr_col = f"{algorithm}_FPR"
    fn_col = f"{algorithm}_FN"
    fp_col = f"{algorithm}_FP"

    # divide group
    protected_group = df['Group'] == f'{label}'
    unprotected_group = ~protected_group

    # test dataset
    protected_tpr = df.loc[protected_group, tpr_col].values
    unprotected_tpr = df.loc[unprotected_group, tpr_col].values

    protected_fpr = df.loc[protected_group, fpr_col].values
    unprotected_fpr = df.loc[unprotected_group, fpr_col].values

    protected_ratio_fn_fp = (df.loc[protected_group, fn_col] / (df.loc[protected_group, fp_col] + eps)).values
    unprotected_ratio_fn_fp = (df.loc[unprotected_group, fn_col] / (df.loc[unprotected_group, fp_col] + eps)).values

    # Normality test
    protected_tpr_n = is_normality(protected_tpr)
    unprotected_tpr_n = is_normality(unprotected_tpr)
    
    if protected_tpr_n > 0.05 and unprotected_tpr_n > 0.05:

        # Homogeneity of Variances Test
        _, p_value_tpr = levene(protected_tpr, unprotected_tpr)  

        if p_value_tpr < 0.05:
            tpr_ttest = ttest_ind(protected_tpr, unprotected_tpr, equal_var=False)
        else:
            tpr_ttest = ttest_ind(protected_tpr, unprotected_tpr, equal_var=True)
    
    else:

        # Non-parametric test
        tpr_ttest = mannwhitneyu(protected_tpr, unprotected_tpr)

    # Normality test
    protected_fpr_n = is_normality(protected_fpr)
    unprotected_fpr_n = is_normality(unprotected_fpr)
    
    if protected_fpr_n > 0.05 and unprotected_fpr_n > 0.05:
        
        # Homogeneity of Variances Test
        _, p_value_fpr = levene(protected_fpr, unprotected_fpr)
        if p_value_fpr < 0.05:
            fpr_ttest = ttest_ind(protected_fpr, unprotected_fpr, equal_var=False)
        else:
            fpr_ttest = ttest_ind(protected_fpr, unprotected_fpr, equal_var=True)
    
    else:

        # Non-parametric test
        fpr_ttest = mannwhitneyu(protected_fpr, unprotected_fpr)

    # Normality test
    protected_ratio_fn_fp_n = is_normality(protected_ratio_fn_fp)
    unprotected_ratio_fn_fp_n = is_normality(unprotected_ratio_fn_fp)

    if protected_ratio_fn_fp_n > 0.05 and unprotected_ratio_fn_fp_n > 0.05:
        
        # Homogeneity of Variances Test
        _, p_value_ratio_fn_fp = levene(protected_ratio_fn_fp, unprotected_ratio_fn_fp)
        if p_value_ratio_fn_fp < 0.05:
            ratio_ttest = ttest_ind(protected_ratio_fn_fp, unprotected_ratio_fn_fp, equal_var=False)
        else:
            ratio_ttest = ttest_ind(protected_ratio_fn_fp, unprotected_ratio_fn_fp, equal_var=True)

    else:

        # Non-parametric test
        ratio_ttest = mannwhitneyu(protected_ratio_fn_fp, unprotected_ratio_fn_fp)


    print(f'{algorithm} -TPR:', tpr_ttest)
    print(f"{algorithm} - FPR:", fpr_ttest)
    print(f"{algorithm} - FN/FP:", ratio_ttest)



def perform_anova_tests(df, algorithm, *labels):

    eps = 1e-10

    tpr_col = f"{algorithm}_TPR"
    fpr_col = f"{algorithm}_FPR"
    fn_col = f"{algorithm}_FN"
    fp_col = f"{algorithm}_FP"

    # divide group
    groups = []
    for label in labels:
        groups.append(df['Group'] == f'{label}')
    
    tpr = []
    for group in groups:
        tpr.append(df.loc[group, tpr_col].values)

    fpr = []
    for group in groups:
        fpr.append(df.loc[group, fpr_col].values)
    
    ratio_fn_fp = []
    for group in groups:
        ratio_fn_fp.append((df.loc[group, fn_col] / (df.loc[group, fp_col] + eps)).values)


    # Normality test
    normal_tpr = []
    for i in tpr:
        normal_tpr.append(is_normality(i))
    
    normal_tpr = np.array(normal_tpr)
    is_normal_tpr = np.all(normal_tpr > 0.05)
    
    if is_normal_tpr:
        # Homogeneity of Variances Test
        _, p_value_tpr = levene(*tpr)

        if p_value_tpr < 0.05:
            tpr_anova = welch_anova(data = df, dv = f"{algorithm}_TPR", between='Group')
        else:
            tpr_anova = f_oneway(*tpr)
    
    else:
        # Non-parametric test
        try:
            tpr_anova = kruskal(*tpr)
        except ValueError:
            print('tpr: ', tpr)
            tpr_anova = -1


    # Normality test
    normal_fpr = []
    for i in fpr:
        normal_fpr.append(is_normality(i))
    
    normal_fpr = np.array(normal_fpr)
    is_normal_fpr = np.all(normal_fpr > 0.05)
    
    if is_normal_fpr:
        # Homogeneity of Variances Test
        _, p_value_fpr = levene(*fpr)

        if p_value_fpr < 0.05:
            fpr_anova = welch_anova(data = df, dv = f"{algorithm}_FPR", between='Group')
        else:
            fpr_anova = f_oneway(*fpr)
    
    else:
        # Non-parametric test
        try:
            fpr_anova = kruskal(*fpr)
        except ValueError:
            print('fpr: ', fpr)
            fpr_anova = -1



    name = algorithm + '_ratio_fn_fp'
    df[name] = df[fn_col]/ (df[fp_col] + eps)

    # Normality test3
    normal_ratio_fn_fp = []
    for i in ratio_fn_fp:
        normal_ratio_fn_fp.append(is_normality(i))
    
    normal_ratio_fn_fp = np.array(normal_ratio_fn_fp)
    is_normal_ratio_fn_fp = np.all(normal_ratio_fn_fp > 0.05)
    
    if is_normal_ratio_fn_fp:
        # Homogeneity of Variances Test
        _, p_value_ratio_fn_fp = levene(*ratio_fn_fp)

        if p_value_ratio_fn_fp < 0.05:
            ratio_fn_fp_anova = welch_anova(data = df, dv = f"{algorithm}_ratio_fn_fp", between='Group')
        else:
            ratio_fn_fp_anova = f_oneway(*ratio_fn_fp)
    
    else:
        # Non-parametric test
        try:
            ratio_fn_fp_anova = kruskal(*ratio_fn_fp)
        except ValueError:
            print('ratio_fn_fp: ', ratio_fn_fp)
            ratio_fn_fp_anova = -1



    print(f'{algorithm} -TPR:', tpr_anova)
    print(f'{algorithm} -FPR:', fpr_anova)
    print(f"{algorithm} - FN/FP:", ratio_fn_fp_anova)










def main():

    df = pd.read_excel('d18_result.xlsx')

    label = 'Elderly people aged 65 and above'

    perform_t_tests(df, 'SVM', label)
    perform_t_tests(df, 'DT', label)
    perform_t_tests(df, 'RF', label)
    perform_t_tests(df, 'LR', label)
    perform_t_tests(df, 'KNN', label)
    perform_t_tests(df, 'ANN', label)


if __name__ == '__main__':
    main()
