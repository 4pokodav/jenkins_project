import pandas as pd

url = 'https://raw.githubusercontent.com/4pokodav/jenkins_project/refs/heads/main/Housing.csv'

df = pd.read_csv(url, delimiter=',')

binary_map = {'yes': 1, 'no': 0}
bool_map = {True: 1, False: 0}

df['mainroad'] = df['mainroad'].map(binary_map)
df['guestroom'] = df['guestroom'].map(binary_map)
df['basement'] = df['basement'].map(binary_map)
df['hotwaterheating'] = df['hotwaterheating'].map(binary_map)
df['airconditioning'] = df['airconditioning'].map(binary_map)
df['prefarea'] = df['prefarea'].map(binary_map)

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

df['furnishingstatus_semi-furnished'] = df['furnishingstatus_semi-furnished'].map(bool_map)
df['furnishingstatus_unfurnished'] = df['furnishingstatus_unfurnished'].map(bool_map)

df.to_csv('processed.csv', index=False)
