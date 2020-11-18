import numpy as np                                                                                                                     
import pandas as pd                                                                                                                    
import statsmodels.api as sm                                                                                                           
from sklearn.linear_model import LogisticRegression                                                                                    

# split the data into two samples 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) 

# 

n = 1000                                                                                                                               
x = np.random.normal(size=(n, 3))                                                                                                      
lp = x[:, 0] - x[:, 1] / 2                                                                                                             
pr = 1 / (1 + np.exp(-lp))                                                                                                             
y = (np.random.uniform(size=n) < pr).astype(np.int)                                                                                    
df = pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})                                                               

alpha = 10                                                                                                                             

model1 = sm.GLM.from_formula("y ~ x1 + x2 + x3", family=sm.families.Binomial(),                                                        
            data=df)                                                                                                                   
result1 = model1.fit_regularized(alpha=alpha/n, L1_wt=1)                                                                               

model2 = sm.Logit.from_formula("y ~ x1 + x2 + x3", data=df)                                                                            
result2 = model2.fit_regularized(alpha=alpha)                                                                                          

x0 = np.concatenate((np.ones((n, 1)), x), axis=1)                                                                                      
model3 = LogisticRegression(C=1/alpha, penalty='l1')                                                                                   
result3 = model3.fit(x0, y)                                                                                                            

print(result1.params, result2.params, result3.coef_)   
