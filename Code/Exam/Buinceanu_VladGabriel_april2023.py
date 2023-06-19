"""Solutions File Data & AI 2 High Stake March 31st 2023"""


############################################################################################################
""" Student Identification """
# Name:
# First Name:
# Student number:
#
############################################################################################################
""" Instructions """
# Enter for each question - if relevant- the Python code you wrote to find the answer to the question.
# Important remark: You need to answer the questions in Canvas.
#                   Your code will be evaluated separately on readability, completeness,
#                   correct use of variables & parameters,... and
#                   will NOT be used to find the answers to questions you left blank in Canvas.
#
# Make your Python code for each question as complete as possible
# (if only the code mentioned for a question is executed, it must deliver the required result(s).
# In other words, it does not require code written for previous questions
# -except for reading the data file(s)-.
#
# If you use homemade functions, mention this in your Python code as a comment.
# Do not forget to copy the home made functions into this solutions file (see section Home made Functions)
# or zip you Python file with your homemade functions together with this solutions file before uploading it
#
# Do not forget to upload this solution file when you have finished the high stakes (see Canvas instructions)
#
############################################################################################################
""" Home made fuctions """
# Copy here the home made functions you used
# or indicate here -as comment- that you will upload the Python file with your home made functions
# together with this solutions file (zip them into one file before you upload!).

#upload the functions

############################################################################################################
""" Python code for reading the data set files HighStake01.csv and HighStake01.csv """
import pandas as pd
import os
from functions import *
os.chdir('/Users/vladbuinceanu/Documents/KdG/Data and A.I./Sem 2/Python/Python/Code/Exam/')

df1 = pd.read_csv('HighStake01.csv', sep=',', decimal='.', na_values='UNKNOWN', on_bad_lines='skip')
df2 = pd.read_csv('HighStake02.csv', sep=',', decimal='.', na_values='UNKNOWN', on_bad_lines='skip')

print(df1.isna().sum().sum())
print(df2.isna().sum().sum())

############################################################################################################
"""Question 1"""


############################################################################################################
"""Question 2"""
from IPython.core.display_functions import display
display(df1.visibility)

############################################################################################################
"""Question 3"""

############################################################################################################
"""Question 4"""

############################################################################################################
"""Question 5"""
#display(all_freq(df1.visibility))

############################################################################################################
"""Question 6"""


############################################################################################################
"""Question 7"""
dispersion(df1.weight)

############################################################################################################
"""Question 8"""
display(all_freq(df1.amount))

############################################################################################################
"""Question 9"""
display(df1.province.unique())

############################################################################################################
"""Question 10"""

display(all_freq(pd.cut(df1.measurement4, bins=3)))

############################################################################################################
"""Question 11"""
df2 = df2[(df2.amount.notnull()) & (df2.number.notnull())]



############################################################################################################
"""Question 12"""
display(outlier_boundaries(df1.measurement2))

############################################################################################################
"""Question 13"""
display(df1.timeSeries2.mean())
display(df1.timeSeries2.median())

############################################################################################################
"""Question 14"""


############################################################################################################
"""Question 15"""


############################################################################################################
"""Question 16"""
display(df1.amount.quantile(q=[0.25, 0.5, 0.75]))

############################################################################################################

