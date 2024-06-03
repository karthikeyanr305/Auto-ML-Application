import pandas as pd

# Assuming df is your DataFrame
# Load your dataset into df
df = pd.read_csv('./data/credit_card_transactions/fraudTrain.csv')

# Subset the DataFrame
fraud = df[df['is_fraud'] == 1].sample(n=100, random_state=1)  # 100 rows where isFraud is 1
non_fraud = df[df['is_fraud'] == 0].sample(n=900, random_state=1)  # 900 rows where isFraud is 0

# Concatenate the two subsets
final_df = pd.concat([fraud, non_fraud])

# Shuffle the final dataframe if needed
final_df = final_df.sample(frac=1, random_state=1).reset_index(drop=True)


final_df.to_csv('fraudTest_mini.csv')
# Now final_df contains the desired subset
