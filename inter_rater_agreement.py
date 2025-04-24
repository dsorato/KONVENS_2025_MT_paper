from statsmodels.stats.inter_rater import fleiss_kappa
import pandas as pd
import numpy as np


#each row represents a subject/item rated by 3 raters, each choosing between category A or B
ratings = [
['A', 'A', 'B'],
['B', 'A', 'B'],
['B', 'B', 'A'],
['A', 'A', 'B'],
['A', 'A', 'A'],
['A', 'A', 'B'],
['A', 'A', 'A'],
['B', 'A', 'B'],
['B', 'B', 'B'],
['B', 'A', 'B'],
['A', 'B', 'B'],
['B', 'B', 'A'],
['A', 'B', 'B'],
['B', 'B', 'B'],
['B', 'B', 'B'],
['A', 'B', 'A'],
['A', 'B', 'A'],
['A', 'A', 'B'],
['A', 'B', 'A'],
['B', 'B', 'B'],
['B', 'A', 'B'],
['A', 'B', 'A'],
['A', 'B', 'B'],
['A', 'A', 'B'],
['B', 'A', 'A'],
['A', 'B', 'A']
]

# Matrix with counts of A and B per row
category_counts = []
for row in ratings:
    count_A = row.count('A')
    count_B = row.count('B')
    category_counts.append([count_A, count_B])


df_counts = pd.DataFrame(category_counts, columns=['A', 'B'])

print(df_counts)

kappa_score = fleiss_kappa(df_counts.values, method='fleiss')
print("Fleiss' Kappa:", kappa_score)