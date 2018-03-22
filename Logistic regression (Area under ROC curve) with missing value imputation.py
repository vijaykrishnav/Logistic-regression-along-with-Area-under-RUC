
# coding: utf-8

# In[1]:


#getting libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mlt
get_ipython().magic(u'matplotlib inline')


# In[2]:


#getting file
crd1=pd.read_csv('D:\\V data analytics\\sample dataset\\Dataset by Imarticus\\Data for logistic regression\\R_Module_Day_7.2_Credit_Risk_Train_data.csv')
crd2=pd.read_csv('D:\\V data analytics\\sample dataset\\Dataset by Imarticus\\Data for logistic regression\\R_Module_Day_8.2_Credit_Risk_Validate_data.csv')


# In[11]:


# to do missing value imputation, we will concat both train and validation data

crd=pd.concat([crd1,crd2],axis=0)


# In[4]:


crd.isna().sum()


# In[12]:


#lets reset index just to be sure
crd.reset_index(inplace=True,drop=True)


# In[13]:


crd.head()


# In[14]:


#all good lets check for missing values
crd.isna().sum()


# In[15]:


#missing value of imputation, to get the list of indices of gender missing values we do
genmiss=crd[crd['Gender'].isna()].index.tolist()


# In[16]:


#to find out the mode we do,
crd.Gender.value_counts()


# In[17]:


#lets fill up missing value by 'Male'
crd['Gender'].iloc[genmiss]='Male'


# In[18]:


crd.Gender.isna().sum()


# In[19]:


#great, lets do the missing value imputation for married
crd.Married.value_counts()


# In[20]:


#lets fill up nas by 'Yes'
crd['Married'].iloc[crd[crd['Married'].isna()].index.tolist()]='Yes'


# In[21]:


crd.Married.isna().sum()


# In[22]:


#lets do missing value impputation of dependents
crd.Dependents.value_counts()


# In[29]:


crd.Dependents.isna().sum()


# In[30]:


#lets fill the nas of dependents missing value in case of not married people as zero
# so we need the dataframe where married is no and dependents is na
dmiss1=crd[(crd['Married']=='No') & (crd['Dependents'].isna())].index.tolist()
print(dmiss1)


# In[31]:


#filling it up with zero
crd['Dependents'].iloc[dmiss1]='0'


# In[32]:


crd.Dependents.value_counts()


# In[33]:


crd.Dependents.isna().sum()


# In[28]:


#lets do a crosstab of gender with dependents
pd.crosstab(crd['Gender'],crd['Dependents'].isna())  # out of 16, 15 values corresponds to male


# In[38]:


#lets do the cross tab of dependents and gender
pd.crosstab(crd['Gender'],crd['Dependents'])


# In[39]:


#since most of male have a mode of 0, lets fill the remaining values by 0
crd['Dependents'][(crd['Dependents'].isna())]='0'


# In[40]:


crd.Dependents.isna().sum()


# In[41]:


#missing value impution of selfemployed
crd.Self_Employed.isna().sum()


# In[42]:


crd.Self_Employed.value_counts()


# In[43]:


#replacing nas with 'No'
crd['Self_Employed'].iloc[crd[crd['Self_Employed'].isna()].index.tolist()]='No'


# In[44]:


#mvi of loanamount loanamount_term
#lets compare the missingvalues of loanamount with loanamountterm
pd.crosstab(crd['Loan_Amount_Term'],crd['LoanAmount'].isna())


# In[45]:


#its evident that loan_amount_term pertaining to 360 has highest numbers of nas.
# we will fill the value by the mean of the loanamount corresponding to the loan_amount_term 360.
crd.groupby(crd['Loan_Amount_Term'])['LoanAmount'].mean()


# In[46]:


# lets fill the loanampunt na's in the range of 360 by 144
crd['LoanAmount'].iloc[crd[(crd['Loan_Amount_Term']==360) & (crd['LoanAmount'].isna())].index.tolist()]=144


# In[47]:


#for the rest of nas lets replace the values by 132
crd['LoanAmount'].iloc[crd[crd['LoanAmount'].isna()].index.tolist()]=132


# In[48]:


#missing value imputation of loan_amount_term
crd.Loan_Amount_Term.isnull().sum()


# In[49]:


crd.Loan_Amount_Term.value_counts()


# In[50]:


#lets fill up by mode, ie 360
crd['Loan_Amount_Term'].iloc[crd[crd['Loan_Amount_Term'].isna()].index.tolist()]=360


# In[51]:


#missing value imutation of Credit_history by Logistic regression

crd.Credit_History.isnull().sum()


# In[56]:


#lets separate the dataframe as train and test, lets take all na's in credithistory as test data
crd_testdata=crd.loc[crd['Credit_History'].isna(),:]


# In[58]:


#to get traindata, we will first get incex of all na's of credithistory
crd_credna_indx=crd[crd['Credit_History'].isna()].index.tolist()


# In[61]:


#getting traindata
crd_traindata_index=[x for x in crd.index.tolist() if x not in crd_credna_indx]


# In[62]:


crd_traindata=crd.iloc[crd_traindata_index]


# In[64]:


#Tp do a logistic regression, we will lose all unimo variables and get dummies for all catagorical variables
crd_train1=pd.get_dummies(crd_traindata.drop(['Loan_ID'],axis=1),drop_first=True)


# In[65]:


crd_train1.head()


# In[67]:


#lets do the same for testdata
crd_test1=pd.get_dummies(crd_testdata.drop(['Loan_ID'],axis=1),drop_first=True)


# In[68]:


crd_test1.head()


# In[71]:


#preparing xtrain and y train, xtest and ytest
x_train1=crd_train1.drop(['Credit_History','Loan_Status_Y'],axis=1)
y_train1=crd_train1['Credit_History']


# In[73]:


#similarly for test set
x_test1=crd_test1.drop(['Credit_History','Loan_Status_Y'],axis=1)
y_test1=crd_test1['Credit_History']


# In[74]:


from sklearn.linear_model import LogisticRegression


# In[75]:


m1=LogisticRegression()


# In[76]:


m1.fit(x_train1,y_train1)


# In[77]:


#predicting for xtest
pred1=m1.predict(x_test1)


# In[78]:


print(pred1)


# In[83]:


crd_test1['Credit_History']=pred1


# In[84]:


crd_new=pd.concat([crd_train1,crd_test1],axis=0)


# In[85]:


crd_new.isnull().sum()


# In[86]:


crd_new.shape


# In[87]:


crd.Credit_History.isna().sum()


# In[88]:


crd['Credit_History'].iloc[crd[crd['Credit_History'].isna()].index.tolist()]=pred1


# In[89]:


crd.isnull().sum()


# In[90]:


crd.head()


# In[ ]:


# lets go ahead woth logistic regression now


# In[97]:


#some basic eda's first
sns.barplot(x=crd['Loan_Status'],y=crd['LoanAmount'],hue=crd['Gender'],data=crd)


# In[98]:


sns.barplot(x=crd['Credit_History'],y=crd['Loan_Status'],hue=crd['Married'],data=crd)


# In[105]:


sns.barplot(x=crd['Loan_Status'],y=crd['ApplicantIncome'],hue=crd['Property_Area'],data=crd) #(so rural property got rejected most)


# In[104]:


sns.barplot(x=crd['Loan_Status'],y=crd['ApplicantIncome'],hue=crd['Dependents'],data=crd) #interesting fact is applicant with..
#..3+ dependents were rejected also were given higher no of loans!!


# In[ ]:


# lets split the data into train and validation,just like how it was given previously


# In[111]:


crd_train=crd.head(len(crd1))


# In[113]:


crd_train.to_csv('D:\\V data analytics\\sample dataset\\Dataset by Imarticus\\Data for logistic regression\\Loan_eligibility_estimation_traindata_withmv_imputed\\crd_train.csv')


# In[114]:


crd_val=crd.tail(len(crd2))


# In[115]:


crd_val.to_csv('D:\\V data analytics\\sample dataset\\Dataset by Imarticus\\Data for logistic regression\\Loan_eligibility_estimation_train _test_data_withmv_imputed\\crd_val.csv')


# ### Logistic regression with area under roc curve

# In[117]:


#getting dummies for all catagorical variables
crd.info()


# In[120]:


#seperating dvb
crd_new1=pd.get_dummies(crd.drop(['Loan_ID','Loan_Status'],axis=1),drop_first=True)


# In[121]:


crd_new1.head()


# In[122]:


x_train=crd_new1


# In[123]:


y_train=crd['Loan_Status']


# In[124]:


m2=LogisticRegression()


# In[125]:


m2.fit(x_train,y_train)


# In[128]:


m2.score(x_train,y_train) 


# In[129]:


# PREDICTING DATA FOR VALIDATION DATA
crd_val.shape


# In[130]:


crd_val.head()


# In[132]:


#preparing data to predict
crd_val1=crd_val.drop(['Loan_ID','Loan_Status'],axis=1)


# In[133]:


#getting dummyvalues for cvs
crdval=pd.get_dummies(crd_val1,drop_first=True)


# In[134]:


crdval.head()


# In[135]:


#predicting
pred2=m2.predict(crdval)


# In[136]:


#getting crosstab
pd.crosstab(crd_val['Loan_Status'],pred2)


# In[137]:


#to get crosstab we download metrics
from sklearn import metrics


# In[138]:


metrics.confusion_matrix(crd_val['Loan_Status'],pred2) #we can interpret the results as predicted
                                                                                #       N      Y
                                                  #          actual loanstatus  N      55      22
                                                  #                             Y      1       289


# In[ ]:


#sensitivity of the model =tp/tp+fn= 289/289+1=99% highly sensitive
#speificity of the model = tn/tn+fp=55/55+22=55/77= 71% speific


# In[140]:


#lets get the valuecount of loanstatus in validation data
crd_val.Loan_Status.value_counts()


# In[141]:


#accuracy score
metrics.accuracy_score(crd_val['Loan_Status'],pred2) #93% great 


# In[172]:


crd_val.head()


# In[179]:


crdval.head()


# In[180]:


crdval['Loan_Status']=crd_val['Loan_Status'] #in crdval, we add loanstatus, since we have to conver it to digits by mapping


# In[181]:


crdval.head()


# In[182]:


crdval['Loan_Status_digit']=crdval.Loan_Status.map({'N':0,'Y':1})


# In[184]:


crdval1=crdval.drop(['Loan_Status'],axis=1)


# In[186]:


crdval1.info()


# In[187]:


crdval.info()


# In[196]:


probvalues_of_dv=m2.predict_proba(crdval1.drop(['Loan_Status_digit'],axis=1))


# In[197]:


Loan_status_prob0=m2.predict_proba(crdval1.drop(['Loan_Status_digit'],axis=1))[:,1]


# In[199]:


#graph
sns.distplot(Loan_status_prob0,kde=False,bins=50)


# In[200]:


#roc curve
fpr,tpr,thresholds=metrics.roc_curve(crd_val['Loan_Status_digit'],Loan_status_prob0)


# In[202]:


mlt.plot(fpr,tpr)


# In[203]:


#getting area under aoc
metrics.roc_auc_score(crd_val['Loan_Status_digit'],Loan_status_prob0)

