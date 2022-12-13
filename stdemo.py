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
st.markdown("Income level 1: under $10k")
st.markdown("Income level 2: 10 to under $20,000")
st.markdown("Income level 3: 20 to under $30,000  ")
st.markdown("Income level 4: 30 to under $40,000")
st.markdown("Income level 5: 40 to under $50,000")
st.markdown("Income level 6: 50 to under $75,000")
st.markdown("Income level 7: 75 to under $100,000")
st.markdown("Income level 8: 100 to under $150,000")
st.markdown("Income level 9: $150,000 or more")
income_level = st.selectbox(label="Income level",
options=(1, 2, 3, 4, 5, 6, 7, 8, 9))
st.markdown("Education level 1: Less than high school (Grades 1-8 or no formal schooling)")
st.markdown("Education level 2: High school incomplete (Grades 9-11 or Grade 12 with NO diploma)")
st.markdown("Education level 3: High school graduate (Grade 12 with diploma or GED certificate)")
st.markdown("Education level 4: Some college, no degree (includes some community college)")
st.markdown("Education level 5: Two-year associate degree from a college or university")
st.markdown("Education level 6: Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)")
st.markdown("Education level 7: Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)")
st.markdown("Education level 8: Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")
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

