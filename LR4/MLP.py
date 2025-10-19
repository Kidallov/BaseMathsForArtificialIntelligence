import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка датасета
data = load_breast_cancer()
X, y = data.data, data.target

# Разделение данных
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Стандартизация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


class MLP:
    def __init__(self, layers, learning_rate=0.01, momentum=0.9, l2_reg=0.01):
        self.layers = layers  # список размеров слоев input, hidden1, hidden2,..., output
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []
        self.velocity = []

        # Инициализация весов
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]) * 0.01)
            self.biases.append(np.zeros((1, layers[i])))
            self.velocity.append(np.zeros_like(self.weights[-1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        activations = [X]
        zs = []

        for i in range(len(self.layers) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i < len(self.layers) - 2:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)

        return activations, zs

    def backward(self, activations, zs, y):
        m = y.shape[0]
        deltas = []

        # Ошибка на выходном слое
        output = activations[-1]
        error = output - y
        delta = error * self.sigmoid_derivative(output)
        deltas.append(delta)

        # Обратное распространение ошибки
        for i in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.relu_derivative(zs[i - 1])
            deltas.append(delta)

        deltas.reverse()

        # Обновление весов
        for i in range(len(self.layers) - 1):
            dw = np.dot(activations[i].T, deltas[i]) / m + self.l2_reg * self.weights[i]
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Обновление с momentum
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * dw

# Обучение модели
mlp = MLP(layers=[30, 64, 32, 1], learning_rate=0.01, momentum=0.9, l2_reg=0.001)
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    activations, zs = mlp.forward(X_train)

    # Backward pass
    mlp.backward(activations, zs, y_train.reshape(-1, 1))

    # Вывод логов (например, каждые 100 эпох)
    if epoch % 100 == 0:
        train_pred = activations[-1]
        train_loss = np.mean((train_pred - y_train.reshape(-1, 1)) ** 2)
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}")

# Тестирование
_, test_activations = mlp.forward(X_test)
test_pred = (test_activations[-1] > 0.5).astype(int)
accuracy = np.mean(test_pred.flatten() == y_test)
print(f"Test Accuracy: {accuracy:.4f}")