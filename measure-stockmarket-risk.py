#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge


# In[3]:


#Load Data 
csv_data = pd.read_csv("Data.csv")
display(csv_data)


# In[4]:


#Show Descriptive Statistics of the Stocks
stocks_describe = csv_data[["BCS","ULVR"]]
display(stocks_describe.describe())


# In[5]:


#Calculate Daily Stocks’ Interest Returns 
csv_data.sort_index(ascending=False, inplace=True) #[dates are decreasing]
csv_data["BCS_Return"] = csv_data["BCS"].pct_change()
csv_data["ULVR_Return"] = csv_data["ULVR"].pct_change()
stock_daily_interest_returns = csv_data.dropna(inplace=True)
stock_daily_interest_returns = csv_data[["Date", "BCS", "BCS_Return", "ULVR", "ULVR_Return"]]
display(stock_daily_interest_returns) 


# In[6]:


#Build a Dataframe of Return Interests of Stocks with [ Rf, Rm, SML, HML, UMD ] and Show the Head of the Dataframe.
return_interest_df = csv_data[["Date", "BCS_Return", "ULVR_Return", "Rf", "Rm", "SMB", "HML", "UMD"]]
display(return_interest_df.head()) 


# In[7]:


#CAPM model for BCS Stock
return_interest_df["R-BCS_Rf"] = return_interest_df["BCS_Return"] - return_interest_df["Rf"]
return_interest_df['Rm_Rf'] = return_interest_df['Rm'] - return_interest_df['Rf']
y = return_interest_df["R-BCS_Rf"]
X = return_interest_df["Rm_Rf"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X1_train = sm.add_constant(X_train)
bcs_capm_reg = sm.OLS(y_train,X1_train).fit()
print(bcs_capm_reg.summary())

#Gets coefficient value for Rm_Rf which is the beta value
Rm_RF_coef_bcs = bcs_capm_reg.params['Rm_Rf']
print(f"BETA value for BCS Stock : {Rm_RF_coef_bcs:.4f}")

#Gets R-squared value for CAPM BCS STOCK
bcs_capm_r2score = bcs_capm_reg.rsquared
print(f"The R-squared value for BCS : {bcs_capm_r2score:.3f}")

#Gets the Adj. R-squared value for CAPM BCS STOCK
capm_bcs_adj = bcs_capm_reg.rsquared_adj
print(f"The  Adj. R-squared value for BCS : {capm_bcs_adj:.3f}")


# In[8]:


#CAPM model for ULVR Stock
return_interest_df["R-ULVR_Rf"] = return_interest_df["ULVR_Return"] - return_interest_df["Rf"]
return_interest_df['Rm_Rf'] = return_interest_df['Rm'] - return_interest_df['Rf']
y = return_interest_df["R-ULVR_Rf"]
X = return_interest_df["Rm_Rf"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X1_train =sm.add_constant(X_train)
ulvr_capm_reg = sm.OLS(y_train,X1_train).fit()
print(ulvr_capm_reg.summary())

#Gets coefficient value for Rm_Rf which is the beta value
Rm_RF_coef_ulvr = ulvr_capm_reg.params["Rm_Rf"]
print(f"BETA value for ULVR Stock : {Rm_RF_coef_ulvr:.4f}")

#Gets R-squared value for CAPM ULVR STOCK
ulvr_capm_r2score = ulvr_capm_reg.rsquared
print(f"The R-squared value for ULVR : {ulvr_capm_r2score:.3f}")

#Gets the Adj. R-squared value for CAPM ULVR STOCK
capm_ulvr_adj = ulvr_capm_reg.rsquared_adj
print(f"The Adj. R-squared value for ULVR : {capm_ulvr_adj:.3f}")


# In[9]:


#Test betas and R2 for ULVR and BCS

print("ULVR:")
if Rm_RF_coef_ulvr > 1:
    print(f" ULVR stock with Beta value : {Rm_RF_coef_ulvr:.4f} is expected to perform best in an up-ward market")
elif Rm_RF_coef_ulvr < 1:
    print(f"ULVR stock with Beta value : {Rm_RF_coef_ulvr:.4f} hold their value best in a down-ward market")
else:
    print(f"ULVR stock with Beta value :  {Rm_RF_coef_ulvr:.4f} is moving in a neutral market")
    
    
if  ulvr_capm_r2score > 0.5:
    print(f"ULVR R-squared score : {ulvr_capm_r2score:.4f} indicates a good fit for the model")
else:
    print(f"ULVR R-squared score : {ulvr_capm_r2score:.4f} indicates a poor fit for the model")
    

print("\nBCS:")
if Rm_RF_coef_bcs > 1:
    print(f"BCS stock with Beta value : {Rm_RF_coef_bcs:.4f} is expected to perform best in an up-ward market")
elif Rm_RF_coef_bcs < 1:
    print(f"BCS stock with Beta value : {Rm_RF_coef_bcs:.4f} hold their value best in a down-ward market")
else:
    print(f"BCS stock with Beta value : {Rm_RF_coef_bcs:.4f} is moving in a neutral market")

    
if  bcs_capm_r2score > 0.5:
    print(f"The R2 value : {bcs_capm_r2score:.4f} indicates a good fit for the model")
else:
    print(f"The R2 value : {bcs_capm_r2score:.4f} stock indicates a poor fit for the model")


# In[10]:


#F&F model for BCS Stock
return_interest_df["R-BCS_Rf"] = return_interest_df["BCS_Return"] - return_interest_df["Rf"]
return_interest_df['Rm_Rf']= return_interest_df['Rm'] - return_interest_df['Rf']
y = return_interest_df["R-BCS_Rf"]
X = return_interest_df[["Rm_Rf","SMB","HML","UMD"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X1_train = sm.add_constant(X_train)
bcs_ff_reg = sm.OLS(y_train,X1_train).fit()
print(bcs_ff_reg.summary())

#Gets coefficient value for Rm_Rf which is the beta value!!
Rm_RF_coef_bcs = bcs_ff_reg.params['Rm_Rf']
print(f"BETA value for BCS Stock : {Rm_RF_coef_bcs:.4f}")

#Gets R-squared value for F&F BCS STOCK
bcs_ff_r2score = bcs_ff_reg.rsquared
print(f"The R-squared value for BCS : {bcs_ff_r2score:.3f}")

#Gets the Adj. R-squared value for F&F BCS Stock
ff_bcs_adj = bcs_ff_reg.rsquared_adj
print(f"The Adj. R-squared value for BCS : {ff_bcs_adj:.3f}")


# In[11]:


#F&F model for ULVR Stock
return_interest_df["R-ULVR_Rf"] = return_interest_df["ULVR_Return"] - return_interest_df["Rf"]
return_interest_df['Rm_Rf'] = return_interest_df['Rm'] - return_interest_df['Rf']
y = return_interest_df["R-ULVR_Rf"]
X = return_interest_df[["Rm_Rf","SMB","HML","UMD"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X1_train =sm.add_constant(X_train)
ulvr_ff_reg = sm.OLS(y_train,X1_train).fit()
print(ulvr_ff_reg.summary())

#Gets coefficient value for Rm_Rf which is the beta value!!
Rm_RF_coef_ulvr = ulvr_ff_reg.params['Rm_Rf']
print(f"BETA value for ULVR Stock is : {Rm_RF_coef_ulvr:.4f}")

#Gets R-squared value for F&F ULVR STOCK
ulvr_ff_r2score = ulvr_ff_reg.rsquared
print(f"The R-squared value for ULVR : {ulvr_ff_r2score:.3f}")

#Gets the Adj. R-squared value for F&F ULVR Stock 
ff_ulvr_adj = ulvr_ff_reg.rsquared_adj
print(f"The  Adj. R-squared value for ULVR : {ff_ulvr_adj:.3f}")


# In[12]:


# Compare the adjusted squared values for BCS Stock
if capm_bcs_adj > ff_bcs_adj:
    bcs_better_model_adj = "CAPM Model"
else:
    bcs_better_model_adj = "F&F Model"

# Compare the adjusted squared values for ULVR Stock
if capm_ulvr_adj > ff_ulvr_adj:
    ulvr_better_model_adj = "CAPM Model"
else:
    ulvr_better_model_adj = "F&F Model"

# Compare the R-squared values for BCS Stock
if bcs_capm_r2score > bcs_ff_r2score:
    bcs_better_model_r2 = "CAPM Model"
else:
    bcs_better_model_r2 = "F&F Model"

# Compare the R-squared values for ULVR Stock
if ulvr_capm_r2score > ulvr_ff_r2score:
    ulvr_better_model_r2 = "CAPM Model"
else:
    ulvr_better_model_r2 = "F&F Model"

# Print the results
print(f"The Better Model for BCS Stock based on the Adjusted R-squared value is {bcs_better_model_adj}")
print(f"The Better Model for ULVR Stock based on the Adjusted R-squared value is {ulvr_better_model_adj}")
print(f"The Better Model for BCS Stock based on the R-squared value is {bcs_better_model_r2}")
print(f"The Better Model for ULVR Stock based on the R-squared value is {ulvr_better_model_r2}")


# In[13]:


#Based on the better model of each stock, calculate the residual between predictions and real values of return interests.

#BCS Stock based on Adjusted R-squared score as f&f is the better model 
X1_test=sm.add_constant(X_test)
predicted_y = bcs_ff_reg.predict(X1_test)
stock_residuals_bcs = pd.DataFrame({'Predicted':predicted_y,'Actual':y_test, 'Residuals':y_test-predicted_y})
display(stock_residuals_bcs.head(10))

#plots histogram
residuals = y_test - predicted_y 
ax = pd.DataFrame({'Residuals':residuals}).hist(bins=25) 
plt.tight_layout() 
plt.show() 


# In[14]:


#ULVR Stock based on Adjusted R-squared score as f&f is the better model 
X1_test = sm.add_constant(X_test)
predicted_y = ulvr_ff_reg.predict(X1_test)
compare = pd.DataFrame({'Predicted':predicted_y,'Actual':y_test, 'Residuals':y_test-predicted_y})
display(compare.head(10))

#Plots histogram
residuals = y_test - predicted_y 
ax = pd.DataFrame({'Residuals':residuals}).hist(bins=25) 
plt.tight_layout() 
plt.show() 

