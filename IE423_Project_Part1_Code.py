#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing our data & libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"/Users/xxx/Desktop/modified_data.csv")
df.sort_values(by=['timestamp'])

#choosing 6 stocks
#print(df.columns.values)
#choose AKBNK,CCOLA,FROTO,PGSUS,THYAO,VESTL

#seeing if AKBNK has enough data
akbnk_timestampf = df.loc[df['AKBNK'].notna()]['timestamp'].iloc[0]
akbnk_timestampl = df.loc[df['AKBNK'].notna()]['timestamp'].iloc[-1]
print("akbnk start:",akbnk_timestampf)
print("akbnk finish:",akbnk_timestampl)

#seeing if CCOLA has enough data
ccola_timestampf = df.loc[df['CCOLA'].notna()]['timestamp'].iloc[0]
ccola_timestampl = df.loc[df['CCOLA'].notna()]['timestamp'].iloc[-1]
print("ccola start:",ccola_timestampf)
print("ccola finish:",ccola_timestampl)

#seeing if FROTO has enough data
froto_timestampf = df.loc[df['FROTO'].notna()]['timestamp'].iloc[0]
froto_timestampl = df.loc[df['FROTO'].notna()]['timestamp'].iloc[-1]
print("froto start:",froto_timestampf)
print("froto finish:",froto_timestampl)

#seeing if PGSUS has enough data
pgsus_timestampf = df.loc[df['PGSUS'].notna()]['timestamp'].iloc[0]
pgsus_timestampl = df.loc[df['PGSUS'].notna()]['timestamp'].iloc[-1]
print("pgsus start:",pgsus_timestampf)
print("pgsus finish:",pgsus_timestampl)

#seeing if THYAO has enough data
thyao_timestampf = df.loc[df['THYAO'].notna()]['timestamp'].iloc[0]
thyao_timestampl = df.loc[df['THYAO'].notna()]['timestamp'].iloc[-1]
print("thyao start:",thyao_timestampf)
print("thyao finish:",thyao_timestampl)

#seeing if VESTL has enough data
vestl_timestampf = df.loc[df['VESTL'].notna()]['timestamp'].iloc[0]
vestl_timestampl = df.loc[df['VESTL'].notna()]['timestamp'].iloc[-1]
print("vestl start:",vestl_timestampf)
print("vestl finish:",vestl_timestampl)


# In[ ]:


df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp_2'].dt.year
df['month'] = df['timestamp_2'].dt.month
df['day'] = df['timestamp_2'].dt.day
#chosen_indices = ['AKBNK', 'CCOLA', 'FROTO', 'PGSUS', 'THYAO', 'VESTL']

print(df)


# In[ ]:


# List of companies
companies = ['AKBNK', 'CCOLA', 'FROTO', 'PGSUS', 'THYAO', 'VESTL']

# Loop through companies and months
for company in companies:
    for year in range(2017, 2020):  # We want to cover 3 years
        for month in range(1, 13):  # For all months
            # Filter the data
            company_data = df[(df['year'] == year) & (df['month'] == month)][[company]]
            
            if not company_data.empty:
                # Create a boxplot
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=company_data, orient="v", color='lightblue')
                plt.title(f'{company} Stock Prices in {month:02d}/{year}')
                plt.ylabel(f'{company} Stock Price')
                plt.show()


# In[ ]:


# Define a multiplier for the 3-sigma rule
sigma_multiplier = 3

# Create a new column to flag outliers
for company in companies:
    df[f'{company}_outlier'] = False  # Initialize the column as False

    for year in range(2017, 2020):  # We want to cover 3 years
        for month in range(1, 13):  # For all months
            # Filter the data for the specific company, year, and month
            data_subset = df[(df['year'] == year) & (df['month'] == month)][[company]]

            if not data_subset.empty:
                # Calculate the mean and standard deviation for the subset
                subset_mean = data_subset[company].mean()
                subset_std = data_subset[company].std()

                # Identify and flag outliers based on the 3-sigma rule
                lower_bound = subset_mean - sigma_multiplier * subset_std
                upper_bound = subset_mean + sigma_multiplier * subset_std
                # Define the condition for which rows will be considered as outliers
                outlier_condition = (
                    (df['year'] == year) & 
                    (df['month'] == month) & 
                    ((df[company] < lower_bound) | (df[company] > upper_bound))
                    )

                # Get the indices of rows that meet the outlier condition
                outlier_indices = df.loc[outlier_condition].index

                # Update the '_outlier' column to True for these indices
                df.loc[outlier_indices, f'{company}_outlier'] = True
                #df.loc[(df['year'] == year) & (df['month'] == month) & (
                    #(df[company] < lower_bound) | (df[company] > upper_bound)), f'{company}_outlier'] = True

# After running the loop, we will have a column for each company indicating whether each data point is an outlier.


# In[ ]:


# Extract the outliers
outliers_akbnk = df[df['AKBNK_outlier']]
outliers_ccola = df[df['CCOLA_outlier']]
outliers_froto = df[df['FROTO_outlier']]
outliers_pgsus = df[df['PGSUS_outlier']]
outliers_thyao = df[df['THYAO_outlier']]
outliers_vestl = df[df['VESTL_outlier']]


# In[ ]:


from IPython.display import display
Akbank_data = outliers_akbnk[['year', 'month', 'day', 'timestamp', 'AKBNK']]
display(Akbank_data)


# In[ ]:


# Filtering the data to include only dates between 2017 and 2019
filtered_df = df[(df['timestamp_2'] >= pd.to_datetime('2017-01-01')) & 
                 (df['timestamp_2'] <= pd.to_datetime('2019-12-31'))]

# Plotting akbank's stock price trend with outliers for each month
company = 'AKBNK'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()


# In[ ]:


from IPython.display import display
ccola_data = outliers_ccola[['year', 'month', 'day', 'timestamp', 'CCOLA']]
display(ccola_data)


# In[ ]:


# Plotting coca cola's stock price trend with outliers for each month
company = 'CCOLA'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()


# In[ ]:


from IPython.display import display
froto_data = outliers_froto[['year', 'month', 'day', 'timestamp', 'FROTO']]
display(froto_data)


# In[ ]:


# Plotting ford's stock price trend with outliers for each month
company = 'FROTO'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()


# In[ ]:


from IPython.display import display
pgsus_data = outliers_pgsus[['year', 'month', 'day', 'timestamp', 'PGSUS']]
pd.set_option('display.max_rows', None)
display(pgsus_data)


# In[ ]:


# Plotting pegasus's stock price trend with outliers for each month
company = 'PGSUS'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()


# In[ ]:


from IPython.display import display
thyao_data = outliers_thyao[['year', 'month', 'day', 'timestamp', 'THYAO']]
display(thyao_data)


# In[ ]:


# Plotting thy's stock price trend with outliers for each month
company = 'THYAO'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()


# In[ ]:


from IPython.display import display
vestl_data = outliers_vestl[['year', 'month', 'day', 'timestamp', 'VESTL']]
display(vestl_data)


# In[ ]:


# Plotting vestel's stock price trend with outliers for each month
company = 'VESTL'
for year in range(2017, 2020):  # Years 2017 to 2019
    for month in range(1, 13):  # Months January to December
        monthly_data = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month)]
            
        if not monthly_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_data['timestamp_2'], monthly_data[company], label=f'{company} Stock Price', linestyle='-')
                
            # Highlight outliers for the specific month
            monthly_outliers = monthly_data[monthly_data[f'{company}_outlier']]
            plt.scatter(monthly_outliers['timestamp_2'], monthly_outliers[company], color='red', label='Outliers', zorder=5)
                
            plt.title(f'{company} Stock Price Trend - {year}-{month:02d}')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()

