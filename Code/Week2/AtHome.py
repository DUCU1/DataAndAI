import pandas as pd
import os
os.chdir('/Sources')

studenq = pd.read_csv('Questionnaire 22-23.csv', delimiter=';', decimal=',')
df = pd.DataFrame(studenq)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Question 1.a
df[df.columns[5]] = df[df.columns[5]].astype(pd.CategoricalDtype(categories=df[df.columns[5]].unique()))

# Question 1.b
df[df.columns[12]] = df[df.columns[12]].astype(pd.CategoricalDtype(categories=['Not at all', 'Little importance', 'Moderate importance', 'Great importance', 'Very great importance', 'Extreemly important'], ordered=True))


