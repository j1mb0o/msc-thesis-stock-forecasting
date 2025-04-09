import pandas as pd
import tqdm

# data = pd.Series(range(100), index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
# train, test = data.iloc[:80], data.iloc[80:] # Train on first 15, test on last 5 (indices 15-19)

# print(type(train.iloc[:50]), type(test))

# # # context 

# context = train.copy()
# print(type(context))
# horizon = 5
# # 
# # for i in tqdm.tqdm(range(0,test.shape[0], horizon)):
# print(len(context))
# for i in range(0, test.shape[0], horizon):
#     context = pd.concat([context, test[i:i+horizon]])
#     print(len(context))

import numpy as np

x = np.array([[81.12903]])
print(x.squeeze())
 