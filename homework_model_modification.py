import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, log_epoch, RegressionDataset, make_classification_data, accuracy, ClassificationDataset
from sklearn.metrics import confusion_matrix

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def test_linear_regression():
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Добавляем гиперпараметры для настройки силы реуляризации
    l1_lambda = 0.01
    l2_lambda = 0.01
    # Данные для early stopping
    best_avg_loss = float('inf')
    best_model = None
    max_count_fails = 30

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=32)

    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)

            # Считаем L1 и L2 регуляризацию
            l1_reg = sum(torch.abs(param).sum() for param in model.parameters())
            l2_reg = sum((param ** 2).sum() for param in model.parameters())
            loss = criterion(y_pred, batch_y) + l1_reg * l1_lambda + l2_reg * l2_lambda

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        # early stopping
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_X, batch_y in test_dataloader:
                y_pred = model(batch_X)
                test_loss += criterion(y_pred, batch_y).item()
            avg_test_loss = test_loss / len(test_dataloader)

        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss, test_loss=avg_test_loss)
        
        # выбираем лучшую версию модели
        if avg_test_loss < best_avg_loss:
            best_avg_loss = avg_test_loss
            best_model = model.state_dict()
        else:
            max_count_fails -= 1
            if max_count_fails == 0:
                print(f'Исчерпано количество измерений. Лучшая модель с loss: {best_avg_loss:.4f}')
                break

    # Сохраняем модель
    torch.save(best_model, 'linreg_torch.pth')
    
    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval() 


def test_logistic_regression():
    # Генерируем данные
    X, y = make_classification_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Данные для confusion matrix
    all_y_pred = []
    all_y_true = []
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            # Вычисляем accuracy
            y_pred = torch.sigmoid(logits) >= 0.5
            acc = accuracy(y_pred, batch_y)

            # Сохраняем значения для создания confusion matrix
            all_y_pred.extend(map(lambda t: t.item(), y_pred))
            all_y_true.extend(map(lambda t: t.item(), batch_y))
            
            total_loss += loss.item()
            total_acc += acc
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    # Создаем confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'logreg_torch.pth')
    
    # Загружаем модель
    new_model = LogisticRegression(in_features=2)
    new_model.load_state_dict(torch.load('logreg_torch.pth'))
    new_model.eval() 


#test_linear_regression()
test_logistic_regression()
