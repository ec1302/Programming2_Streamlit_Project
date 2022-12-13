import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Original data
s = pd.read_csv("social_media_usage.csv")
def clean_sm(x):
    x = np.where(x == 1,1,0)
    return x

# Creating a dataframe of what we need
ss = pd.DataFrame({'sm_li':clean_sm(s['web1h']),
'income':np.where(s['income'] > 9, np.nan, s['income']),
'educ':np.where(s['educ2'] > 8, np.nan, s['educ2']),
'parent':np.where(s['par'] == 2, 0, 1),
'married': np.where(s['marital'] == 1, 1, 0),
'female': np.where(s['gender'] == 2, 1, 0 ),
'age': np.where(s['age'] > 98, np.nan, s['age'])})
ss = ss.dropna()

# Machine Learning
y = ss['sm_li']
X = ss.drop('sm_li', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,      
                                                    test_size = 0.2,    
                                                    random_state = 123) 

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X_train, y_train)


# Application interface and interactions

# Introductory graphs
st.markdown("# Welcome to my programming project")
st.markdown("##### This is the data we will be using:")
if st.checkbox("Show data", key="data"):
    st.dataframe(ss)
data1 = ss.groupby(['age', 'female'], as_index = False)['sm_li'].mean()
alt_plot1 = alt.Chart(data1).mark_line().encode(
    x = 'age', 
    y = 'sm_li', 
    color = 'female:N')
st.markdown('##### Here is a graph showing the proportion of users who use LinkedIn based on age and gender:')
if st.checkbox("Show graph", key="plot1"):
    alt_plot1
data2 = ss.groupby(['income', 'parent'], as_index = False)['sm_li'].mean()
alt_plot2 = alt.Chart(data2).mark_line().encode(
    x = 'income', 
    y = 'sm_li', 
    color = 'parent:N')
st.markdown('##### Here is a graph showing the proportion of users who use LinkedIn based on income level and parental status:')
if st.checkbox("Show graph", key="plot2"):
    alt_plot2
data3 = ss.groupby(['educ', 'married'], as_index = False)['sm_li'].mean()
alt_plot3 = alt.Chart(data3).mark_line().encode(
    x = 'educ', 
    y = 'sm_li', 
    color = 'married:N')
st.markdown('##### Here is a graph showing the proportion of users who use LinkedIn based on education level and marital status:')
if st.checkbox("Show graph", key="plot3"):
    alt_plot3
# Inputs
st.markdown("## We will now predict whether someone uses LinkedIn based on chosen inputs")
income_level = st.selectbox(label="Income level",
options=(1, 2, 3, 4, 5, 6, 7, 8, 9))
education_level = st.selectbox(label="Education level",
options=(1, 2, 3, 4, 5, 6, 7, 8))
parent = st.selectbox(label="Parental status",
options=(0, 1))
married = st.selectbox(label="Marital status",
options=(0, 1))
female = st.selectbox(label="Female",
options=(0, 1))
age = st.slider("Age")

# Machine learning predictions
test_person = pd.DataFrame({'income':[income_level], 'educ':[education_level], 'parent':[parent], 'married':[married], 'female':[female], 'age':[age]})
predicted_class = lr.predict(test_person)
probs = lr.predict_proba(test_person)
st.markdown("###### These are the parameters chosen")
st.write(test_person)
st.markdown("##### Is this person predicted to use LinkedIn? 1 = Yes 0 = No")
st.write(predicted_class[0])
st.markdown("##### What is the predicted probabiilty of this person using LinkedIn?")
st.write(probs[0][1].round(3))

