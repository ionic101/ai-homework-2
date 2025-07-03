# Домашнее задание к уроку 2: Линейная и логистическая регрессия

## Цель задания
Закрепить навыки работы с PyTorch API, изучить модификацию моделей и работу с различными датасетами.

## Задание 1: Модификация существующих моделей

Создайте файл `homework_model_modification.py`:

### 1.1 Расширение линейной регрессии (15 баллов)
Добавление L1 и L2 регуляризации
```python
l1_lambda = 0.01
l2_lambda = 0.01
...
l1_reg = sum(torch.abs(param).sum() for param in model.parameters())
l2_reg = sum((param ** 2).sum() for param in model.parameters())
loss = criterion(y_pred, batch_y) + l1_reg * l1_lambda + l2_reg * l2_lambda
```

Добавление early stopping
```python
test_dataloader = DataLoader(dataset, batch_size=32)
best_avg_loss = float('inf')
best_model = None
max_count_fails = 30
...
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_X, batch_y in test_dataloader:
        y_pred = model(batch_X)
        test_loss += criterion(y_pred, batch_y).item()
    avg_test_loss = test_loss / len(test_dataloader)
...
if avg_test_loss < best_avg_loss:
    best_avg_loss = avg_test_loss
    best_model = model.state_dict()
else:
    max_count_fails -= 1
    if max_count_fails == 0:
        print(f'Исчерпано количество измерений. Лучшая модель с loss: {best_avg_loss:.4f}')
        break
```

### 1.2 Расширение логистической регрессии (15 баллов)
Добавление поддержки многоклассовой классификации
```python
None
```

Реализация метрик: precision, recall, F1-score, ROC-AUC
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_probs)
```

Визуализация confusion matrix
```python
from sklearn.metrics import confusion_matrix
...
# Данные для confusion matrix
all_y_pred = []
all_y_true = []
...
# Сохраняем значения для создания confusion matrix
all_y_pred.extend(map(lambda t: t.item(), y_pred))
all_y_true.extend(map(lambda t: t.item(), batch_y))
...
cm = confusion_matrix(all_y_true, all_y_pred)
print(cm)
```

## Задание 2: Работа с датасетами (30 баллов)

Создайте файл `homework_datasets.py`:

### 2.1 Кастомный Dataset класс (15 баллов)
```python
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
```

### 2.2 Эксперименты с различными датасетами (15 баллов)
```python
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию
```

## Задание 3: Эксперименты и анализ (20 баллов)

Создайте файл `homework_experiments.py`:

### 3.1 Исследование гиперпараметров (10 баллов)
```python
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц
```

### 3.2 Feature Engineering (10 баллов)
```python
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью
```

## Дополнительные требования

1. **Код должен быть модульным** - разделите на функции и классы
2. **Документация** - добавьте подробные комментарии и docstring
3. **Визуализация** - создайте графики для анализа результатов
4. **Тестирование** - добавьте unit-тесты для критических функций
5. **Логирование** - используйте logging для отслеживания процесса обучения

## Структура проекта

```
homework/
├── homework_model_modification.py
├── homework_datasets.py
├── homework_experiments.py
├── data/                    # Датасеты
├── models/                  # Сохраненные модели
├── plots/                   # Графики и визуализации
└── README.md               # Описание решения
```

## Срок сдачи
Домашнее задание должно быть выполнено до начала занятия 4.

## Полезные ссылки
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)

Удачи в выполнении задания! 🚀 