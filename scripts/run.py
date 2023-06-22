from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import f_oneway
from scikit_posthocs import posthoc_tukey


def one_way_anova(*groups):
    # Perform one-way ANOVA
    f_value, p_value = f_oneway(*groups)

    # Calculate SS, DF, and MS
    n_groups = len(groups)
    total_ss = sum(sum(group) ** 2 for group in groups) - sum(sum(group) for group in groups) ** 2 / sum(len(group) for group in groups)
    ss_between = sum(sum(group) ** 2 / len(group) for group in groups) - sum(sum(group) for group in groups) ** 2 / sum(len(group) for group in groups)
    ss_within = total_ss - ss_between
    df_between = n_groups - 1
    df_within = sum(len(group) for group in groups) - n_groups
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Source of Variation': ['Between Groups', 'Within Groups', 'Total'],
        'Sum of Squares (SS)': [ss_between, ss_within, total_ss],
        'Degrees of Freedom (DF)': [df_between, df_within, df_between + df_within],
        'Mean Squares (MS)': [ms_between, ms_within, None],
        'F-value': [f_value, None, None],
        'p-value': [p_value, None, None]
    })

    return results_df


def plot_tukey(data, name):
    ax = sns.boxplot(data=data, x="classifier", y=METRIC)
    fig = plt.gcf()
    fig.set_size_inches(21, 10)
    
    plt.savefig(name, dpi=300)
    plt.close()
    



DATA_PATH = 'data/dados_para_teste.csv'
RESULTS_PATH = 'results'
METRIC = 'Test accuracy'

# Read Data

data = pd.read_csv(path.join(DATA_PATH))

# Save groups
ax = data.plot(x='cnn', y='classifier', kind='scatter')
fig = plt.gcf()
fig.set_size_inches(21, 5)
plt.savefig(path.join(RESULTS_PATH, 'groups.png'), dpi=300)
plt.close()

# Group by feature method
best_comb_list = []
groups_cnn = data.groupby('cnn')
for cnn in groups_cnn.groups.keys():
    # Group by classifier
    grousp_clf = groups_cnn.get_group(cnn).groupby('classifier')
    # Perform ANOVA one-way
    anova_results = one_way_anova(*(grousp_clf.get_group(group)[METRIC] for group in grousp_clf.groups.keys()))
    # Save ANOVA results:
    anova_results.to_csv(path.join(RESULTS_PATH, f'ANOVA_{cnn}.csv'))
    
    # Get the best model within the group:
    sorted_models = groups_cnn.get_group(cnn).groupby('classifier').agg(['mean'])[METRIC, 'mean'].sort_values(ascending=False)
    best_model_name = sorted_models.index[0]
    best_comb_list.append((cnn, best_model_name))
    
    # Save boxplot for TUKEY
    plot_tukey(groups_cnn.get_group(cnn), name=path.join(RESULTS_PATH, f'boxplot-tukey{cnn}.png'))
    

# Filter data
filtered_data = data[data.apply(lambda row: tuple(row[['cnn', 'classifier']]), axis=1).isin(best_comb_list)]
filtered_data.to_csv(path.join(RESULTS_PATH, 'best_combinations.csv'))

# Perform ANOVA one-way
groups_cnn = filtered_data.groupby('cnn')
anova_results = one_way_anova(*(groups_cnn.get_group(group)[METRIC] for group in groups_cnn.groups.keys()))
# Save ANOVA results:
anova_results.to_csv(path.join(RESULTS_PATH, f'overall_ANOVA.csv'))

# Perform Tukey test within the groups
tukey_test = pairwise_tukeyhsd(filtered_data[METRIC], filtered_data['cnn'])
tukey_df = pd.DataFrame(data=tukey_test._results_table.data[1:], columns=tukey_test._results_table.data[0])
    
tukey_df.to_csv(path.join(RESULTS_PATH, 'overall_TUKEY_HSD.csv'))