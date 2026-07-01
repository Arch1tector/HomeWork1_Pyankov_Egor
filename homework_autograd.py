import torch

"""
Домашнее задание №2
Автоматическое дифференцирование
"""

# 2.1 Простые вычисления с градиентами

# Создаем переменные, для которых необходимо вычислять градиенты
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# Вычисляем заданную функцию
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Запускаем автоматическое вычисление градиентов
f.backward()

print("2.1")
print("f =", f.item())
print("df/dx =", x.grad)
print("df/dy =", y.grad)
print("df/dz =", z.grad)

# Вычисляем градиенты вручную для проверки
dx = 2 * x.item() + 2 * y.item() * z.item()
dy = 2 * y.item() + 2 * x.item() * z.item()
dz = 2 * z.item() + 2 * x.item() * y.item()

# Тесты
assert abs(x.grad.item() - dx) < 1e-6
assert abs(y.grad.item() - dy) < 1e-6


# 2.2 Градиент функции MSE

# Создаем входные данные и правильные значения
x = torch.tensor([1., 2., 3., 4.])
y_true = torch.tensor([3., 5., 7., 9.])

# Параметры линейной модели
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Вычисляем предсказания модели
y_pred = w * x + b

# Вычисляем среднеквадратичную ошибку
mse = torch.mean((y_pred - y_true) ** 2)

# Находим градиенты по параметрам модели
mse.backward()

print("2.2")
print("MSE =", mse.item())
print("dw =", w.grad)
print("db =", b.grad)

# Тесты
assert mse.item() >= 0
assert w.grad is not None and b.grad is not None


# 2.3 Проверка цепного правила

# Создаем переменную с вычислением градиента
x = torch.tensor(2.0, requires_grad=True)

# Вычисляем составную функцию
f = torch.sin(x**2 + 1)

# Находим градиент с помощью backward()
f.backward()

# Повторно вычисляем градиент через autograd.grad()
grad = torch.autograd.grad(torch.sin(x**2 + 1), x)[0]

print("2.3")
print("df/dx =", x.grad)
print("Проверка autograd.grad =", grad)

# Тесты
assert x.grad is not None
assert torch.allclose(x.grad, grad)