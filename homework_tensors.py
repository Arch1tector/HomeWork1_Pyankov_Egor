import torch

"""
Домашнее задание №1
Работа с тензорами PyTorch
"""

# 1.1 Создание тензоров

# Создаем тензор 3x4 со случайными числами
random_tensor = torch.rand(3, 4)

# Создаем тензор, заполненный нулями
zeros_tensor = torch.zeros(2, 3, 4)

# Создаем тензор, заполненный единицами
ones_tensor = torch.ones(5, 5)

# Создаем последовательность чисел от 0 до 15 и меняем ее форму на 4x4
range_tensor = torch.arange(16).reshape(4, 4)

print("1.1 Создание тензоров")
print(random_tensor)
print(zeros_tensor)
print(ones_tensor)
print(range_tensor)

# Тесты
assert random_tensor.shape == (3, 4)
assert zeros_tensor.shape == (2, 3, 4)
assert ones_tensor.shape == (5, 5)
assert range_tensor.shape == (4, 4)


# 1.2 Операции с тензорами

# Создаем две матрицы подходящих размеров
A = torch.rand(3, 4)
B = torch.rand(4, 3)

# Транспонируем матрицу A
A_transpose = A.T

# Выполняем матричное умножение
matrix_product = torch.matmul(A, B)

# Поэлементно умножаем A и транспонированную B
elementwise = A * B.T

# Находим сумму всех элементов матрицы A
sum_A = torch.sum(A)

print("1.2 Операции")
print("Транспонированный A:")
print(A_transpose)

print("A * B:")
print(matrix_product)

print("Поэлементное умножение:")
print(elementwise)

print("Сумма элементов A:")
print(sum_A)

# Тесты
assert A_transpose.shape == (4, 3)
assert matrix_product.shape == (3, 3)
assert elementwise.shape == (3, 4)


# 1.3 Индексация и срезы

# Создаем трехмерный тензор размером 5x5x5
tensor = torch.arange(125).reshape(5, 5, 5)

# Извлекаем первую строку из каждой матрицы
first_row = tensor[:, 0, :]

# Извлекаем последний столбец из каждой матрицы
last_column = tensor[:, :, -1]

# Получаем центральную подматрицу размером 2x2
center_matrix = tensor[2, 1:3, 1:3]

# Получаем элементы с четными индексами по всем измерениям
even_indices = tensor[::2, ::2, ::2]

print("1.3 Индексация")
print("Первая строка:")
print(first_row)

print("Последний столбец:")
print(last_column)

print("Подматрица 2x2:")
print(center_matrix)

print("Элементы с четными индексами:")
print(even_indices)

# Тесты
assert tensor.shape == (5, 5, 5)
assert center_matrix.shape == (2, 2)


# 1.4 Работа с формами

# Создаем одномерный тензор из 24 элементов
vector = torch.arange(24)

# Изменяем форму одного и того же тензора разными способами
shape_2x12 = vector.reshape(2, 12)
shape_3x8 = vector.reshape(3, 8)
shape_4x6 = vector.reshape(4, 6)
shape_2x3x4 = vector.reshape(2, 3, 4)
shape_2x2x2x3 = vector.reshape(2, 2, 2, 3)

print("1.4 Работа с формами")
print(shape_2x12)
print(shape_3x8)
print(shape_4x6)
print(shape_2x3x4)
print(shape_2x2x2x3)

# Тесты
assert shape_2x12.shape == (2, 12)
assert shape_2x2x2x3.shape == (2, 2, 2, 3)