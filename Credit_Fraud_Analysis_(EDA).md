## Credit Card Fraud Detection

### Contex

Using anomolized features and data as a reuslt of a PCA transformation (how we were given the data on kaggle https://www.kaggle.com/mlg-ulb/creditcardfraud), we have been tasked to identify and predict future fraudulent charges.




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')

%matplotlib inline
```


```python
data = pd.read_csv('creditcard.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    

Here we can see that the dataset consists of 31 total columns including the time of the transaction, the 28 different metrics returned by the PCA transformation, the amount of the transaction and the classification (our target) - whether the transaction was fraud or not. We have no missing rows of data.


```python
# defining our classes for EDA
fraud = data[data['Class'] == 1]
not_fraud = data[data['Class'] == 0]

print('There are {} fraud cases'.format(len(fraud)))
print('There are {} valid cases'.format(len(not_fraud)))
print('There are {:.2%} more valid cases than fraud cases'.format(
    (1 - len(fraud)/len(not_fraud)) 
    ))
```

    There are 492 fraud cases
    There are 284315 valid cases
    There are 99.83% more valid cases than fraud cases
    

We can see from the number of fraud cases compared to the number of valid transactions we have highly a highly imbalanced set of data. We will have to account for this by different methods of resampling our data. We will explore not resampling first, however due to the PCA transformation the 28 PCA components are arleady scaled however the amount and time our not we will account for this too.

We can use Scikit learns Standard Scaler or Robust scaler which is more robust to outliers. First lets visualize and choose the best method to scale


```python
fig, ax = plt.subplots(1, 2, figsize=(18,5))
sns.distplot(data.Amount, ax=ax[0])
ax[0].set_title('Distribution plot of Transaction Amount')

sns.distplot(data.Time, ax=ax[1])
ax[1].set_title('Distribution of Transaction Time')

```




    Text(0.5, 1.0, 'Distribution of Transaction Time')




    
![png](output_7_1.png)
    


We can see that Amount spent per transcation widely ranges with large outliers compared to the majority of transactions. We can compare the effects of different scaling options


```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

def scaling(scaler, df, int_scaling):
    clf = scaler().fit_transform(df[int_scaling].values.reshape(-1,1))
    return clf
            

```


```python
amount_std_scaled = scaling(StandardScaler, data, 'Amount')
amount_rob_scaled = scaling(RobustScaler, data, 'Amount')
amount_mm_scaled = scaling(MinMaxScaler, data, 'Amount')


fig, ax = plt.subplots(1, 4, figsize = (18,5))
sns.distplot(amount_std_scaled, ax = ax[0])
sns.distplot(amount_rob_scaled, ax = ax[1])
sns.distplot(amount_mm_scaled, ax = ax[2])
sns.distplot(data.Amount, ax=ax[3], color='r')

ax[0].set_title('Standard Scaling of Price of Transaction')
ax[0].set_xlim([min(amount_std_scaled), max(amount_std_scaled)])

ax[1].set_title('Robust Scaling of Price of Transaction')
ax[1].set_xlim([min(amount_rob_scaled), max(amount_rob_scaled)])

ax[2].set_title('Min Max Scaling of Price of Transaction')
ax[2].set_xlim([min(amount_mm_scaled), max(amount_mm_scaled)])

ax[3].set_title('No Scaling of Price of Transaction')
ax[3].set_xlim([min(data.Amount), max(data.Amount)])

```




    (0.0, 25691.16)




    
![png](output_10_1.png)
    


Here we can see the diffent effects of scaling. Min Max is the most agressing by normalizing the amount between 0 and 1, while robust seams to be the most inclusive of outliers/ the larger amount. We will use the robust for scaling amount and time


```python
data['scaled_amount'] =amount_rob_scaled
time_rob_scaled = scaling(RobustScaler, data, 'Time')
data['scaled_time'] = time_rob_scaled
```


```python
fig, ax = plt.subplots(1,2, figsize=(18,5))

sns.distplot(time_rob_scaled, ax = ax[0])
sns.distplot(data.Time, ax=ax[1], color='r')

ax[0].set_title('Robust Scaled Time of Transcastions')
ax[0].set_xlim([min(time_rob_scaled), max(time_rob_scaled)])

ax[1].set_title('Orginial Transaction Times')
ax[1].set_xlim([min(data.Time), max(data.Time)])

```




    (0.0, 172792.0)




    
![png](output_13_1.png)
    


Here we see that scaling the time data does not change the overall form however compreses the time range between -1 and 1 and adjusts the magnitude of the transactons accordingly.


```python
# quick comparisson of scaled/non scaled values

data[['scaled_time','Time','scaled_amount','Amount']].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scaled_time</th>
      <th>Time</th>
      <th>scaled_amount</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.994983</td>
      <td>0.0</td>
      <td>1.783274</td>
      <td>149.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.994983</td>
      <td>0.0</td>
      <td>-0.269825</td>
      <td>2.69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.994972</td>
      <td>1.0</td>
      <td>4.983721</td>
      <td>378.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.994972</td>
      <td>1.0</td>
      <td>1.418291</td>
      <td>123.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.994960</td>
      <td>2.0</td>
      <td>0.670579</td>
      <td>69.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.994960</td>
      <td>2.0</td>
      <td>-0.256131</td>
      <td>3.67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.994937</td>
      <td>4.0</td>
      <td>-0.237686</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.994901</td>
      <td>7.0</td>
      <td>0.262698</td>
      <td>40.80</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.994901</td>
      <td>7.0</td>
      <td>0.994900</td>
      <td>93.20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.994878</td>
      <td>9.0</td>
      <td>-0.255991</td>
      <td>3.68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.994866</td>
      <td>10.0</td>
      <td>-0.198421</td>
      <td>7.80</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.994866</td>
      <td>10.0</td>
      <td>-0.167819</td>
      <td>9.99</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.994866</td>
      <td>10.0</td>
      <td>1.390344</td>
      <td>121.50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.994854</td>
      <td>11.0</td>
      <td>0.076853</td>
      <td>27.50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.994843</td>
      <td>12.0</td>
      <td>0.514218</td>
      <td>58.80</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.994843</td>
      <td>12.0</td>
      <td>-0.083980</td>
      <td>15.99</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.994843</td>
      <td>12.0</td>
      <td>-0.125900</td>
      <td>12.99</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.994831</td>
      <td>13.0</td>
      <td>-0.294977</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.994819</td>
      <td>14.0</td>
      <td>0.346538</td>
      <td>46.80</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.994807</td>
      <td>15.0</td>
      <td>-0.237546</td>
      <td>5.00</td>
    </tr>
  </tbody>
</table>
</div>



### Looking at correlation between our features and classes


```python
corr = data.drop('Time', axis=1).drop('Amount', axis=1).corr()

sns.set(style='white')
fig, ax = plt.subplots(figsize=(9,7))

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 100, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2082ab35430>




    
![png](output_17_1.png)
    



```python
fig, ax = plt.subplots(figsize=(9,7))
corr_bar = data.drop('scaled_amount', axis=1).drop('scaled_time', axis=1).drop('Class', axis =1).corrwith(data.Class).plot(kind='bar', ax =ax)  #creates pairwase correlation
ax.set_title('Correlation of Parameters to Fraud')
fig.savefig('images/corr_plot.png')
```


    
![png](output_18_0.png)
    


From looking at the correlation matrix and also relative correlation with Class, we see some features are negatively correlated (V3, V7, V10,V14, etc..) while (V2, V4, V11) seem to be positevly correlated. We can look into that in their individual distribution plots for further insights.  


```python
fig, ax = plt.subplots(7,5, figsize=(18,30))
i=0 # intilizing counter
for  ax,( i, col) in zip(ax.flat, enumerate(data.columns)):
#     print(ax, i,col)
    sns.boxplot(x= 'Class', y=col, data=data, ax=ax)
#     ax.set_xlim([min[data[col]], max[data[col]]])
    ax.set_title('{} Distribution'.format(col))
    ax.set_xlabel ('')
    ax.set_ylabel('')

```


    
![png](output_20_0.png)
    



```python
fig, ax = plt.subplots(7,5, figsize=(18,30))
i=0 # intilizing counter
for  ax,( i, col) in zip(ax.flat, enumerate(data.columns)):
#     print(ax, i,col)
    sns.violinplot(x= 'Class', y=col, data=data, ax=ax)
#     ax.set_xlim([min[data[col]], max[data[col]]])
    ax.set_title('{} Distribution'.format(col))
    ax.set_xlabel ('')
    ax.set_ylabel('')
```


    
![png](output_21_0.png)
    


From the box plots above we find some interesting insights. Just as in our correlaltion above, the negative correlated features(V3, V7, V10,V14, etc..) and positively (V2, V4, V11) seem to show the a statistal difference in the fraud and non fraud transactions, we could also confirm using ttests and statistal analysis in future work.  


```python
# looking at the most positively correlated parameters
int_param = da
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
      <td>1.783274</td>
      <td>-0.994983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
      <td>-0.269825</td>
      <td>-0.994983</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
      <td>4.983721</td>
      <td>-0.994972</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
      <td>1.418291</td>
      <td>-0.994972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
      <td>0.670579</td>
      <td>-0.994960</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
corr_with = data.drop('scaled_amount', axis=1).drop('scaled_time', axis=1).drop('Class', axis =1).corrwith(data.Class)
pos_corr = corr_with[corr_with >0].sort_values(ascending=False)

int_parameters = pos_corr.head(6).index.values

```


```python
fig, ax = plt.subplots((len(int_parameters)//3) ,3, figsize=(15,10))
for  ax,( i, col) in zip(ax.flat, enumerate(data[int_parameters].columns)):
#     print(ax, i,col)
    sns.violinplot(x= 'Class', y=col, data=data, ax=ax)
#     ax.set_xlim([min[data[col]], max[data[col]]])
    ax.set_title('{} Distribution'.format(col))
    ax.set_xlabel ('')
    ax.set_ylabel('')
fig.tight_layout()
fig.saveimg('positive_corr.png')
fig.savefig('images/positive_corr.png')
```


    
![png](output_26_0.png)
    



```python
fig.savefig('images/positive_corr.png')
```

We see some intersting insights here. The parameters are sorted by the correlation with fraud, we observer large discrepnsies in averages and modes for the top 3 parameters however does not seem as obvious from the others. We can run T-Tests to see if there is any statistal difference between the cases for these parameters. 


```python

```
