import torch
import time

"""
Домашнее задание №3
Сравнение производительности CPU и CUDA
"""


# 3.2 Измерение времени

def measure_cpu(operation):
    """Измеряет время выполнения операции на CPU."""
    # Засекаем время до выполнения операции
    start = time.time()

    # Выполняем переданную операцию
    operation()

    # Засекаем время после выполнения
    end = time.time()

    # Возвращаем время в миллисекундах
    return (end - start) * 1000


def measure_gpu(operation):
    """Измеряет время выполнения операции на GPU."""

    # Если CUDA недоступна, возвращаем None
    if not torch.cuda.is_available():
        return None

    # Создаем события начала и конца измерения
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Начинаем измерение
    start.record()

    # Выполняем операцию
    operation()

    # Завершаем измерение
    end.record()

    # Ждем завершения всех вычислений на GPU
    torch.cuda.synchronize()

    # Возвращаем время выполнения в миллисекундах
    return start.elapsed_time(end)


def speedup(cpu, gpu):
    # Если GPU недоступен, ускорение вычислить нельзя
    if gpu is None:
        return "-"

    # Вычисляем отношение времени CPU к времени GPU
    return f"{cpu / gpu:.2f}x"


# Проверяем, что функции существуют
assert callable(measure_cpu)
assert callable(measure_gpu)


# 3.1 Подготовка данных

# Размеры тестовых тензоров
sizes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256)
]

# Определяем, какое устройство используется
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Используется устройство:", device)


# 3.3 Сравнение операций CPU и CUDA

# Последовательно тестируем каждый размер тензора
for size in sizes:

    print(f"Размер данных: {size}")

    # Создаем случайный тензор на CPU
    cpu_tensor = torch.rand(size)

    # Проверяем корректность созданного тензора
    assert cpu_tensor.shape == size
    assert cpu_tensor.dtype == torch.float32

    # Если GPU доступен, копируем тензор на видеокарту
    if torch.cuda.is_available():
        gpu_tensor = cpu_tensor.to("cuda")

    # Список операций для тестирования
    operations = {
        "MatMul": lambda x: torch.matmul(x, x.transpose(-1, -2)),
        "Add": lambda x: x + x,
        "Multiply": lambda x: x * x,
        "Transpose": lambda x: x.transpose(-1, -2),
        "Sum": lambda x: torch.sum(x)
    }

    # Выводим заголовок таблицы
    print(f"{'Операция':15} {'CPU(ms)':>10} {'GPU(ms)':>10} {'Speedup':>10}")

    # Измеряем время выполнения каждой операции
    for name, operation in operations.items():

        # Время выполнения на CPU
        cpu_time = measure_cpu(lambda: operation(cpu_tensor))

        # Время выполнения на GPU (если доступен)
        if torch.cuda.is_available():
            gpu_time = measure_gpu(lambda: operation(gpu_tensor))
        else:
            gpu_time = None

        # Проверяем корректность измерений
        assert cpu_time >= 0

        if gpu_time is not None:
            assert gpu_time >= 0

        # Выводим результаты измерений
        print(
            f"{name:15}"
            f"{cpu_time:10.2f}"
            f"{gpu_time if gpu_time is not None else '-':>10}"
            f"{speedup(cpu_time, gpu_time):>10}"
        )


# 3.4 Анализ результатов

print("""
1. Наибольшее ускорение обычно показывает матричное умножение,
   поскольку GPU эффективно выполняет большое количество параллельных вычислений.

2. Простые операции (например, транспонирование и вычисление суммы)
   могут быть медленнее на GPU для небольших массивов из-за накладных расходов.

3. Чем больше размер матриц, тем выше потенциальное ускорение,
   так как GPU лучше загружается вычислительной работой.

4. Передача данных между CPU и GPU требует дополнительного времени.
   Для небольших объемов данных время копирования может превысить
   время самих вычислений.
""")