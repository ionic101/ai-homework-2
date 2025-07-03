from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)

    def normalize(self):
        scaler = MinMaxScaler()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
    
    def fill_none(self):
        self.data = self.data.fillna('NaN')
    
    def encode(self):
        self.fill_none()
        categorical_cols = self.data.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
    
    def __str__(self):
        return str(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


if __name__ == '__main__':
    dataset = CustomDataset(
        csv_path='data/titanic_train.csv',
    )
    print(dataset)
    dataset.encode()
    dataset.normalize()
    print(dataset)
