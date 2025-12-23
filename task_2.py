"""
Модуль обчислює визначений інтеграл методом Монте-Карло та перевіряє результат.

Мета:
- Рахуємо площу під графіком функції f(x) на відрізку [a, b] методом Монте-Карло.
- Перевіряємо точність порівнянням з аналітичним інтегралом та (опційно) з scipy.integrate.quad.

Підхід:
- Створюємо прямокутник [a, b] × [0, f_max].
- Генеруємо випадкові точки у прямокутнику.
- Рахуємо частку точок, які потрапили під криву y <= f(x).
- Множимо частку на площу прямокутника — отримуємо оцінку інтеграла.

Примітка:
- Для прикладу використовується f(x)=x^2 на [0, 2].
"""

from __future__ import annotations

import random
from typing import Callable, Tuple


def integrand_function(x_value: float) -> float:
    """Обчислює значення підінтегральної функції f(x)=x^2."""
    return x_value ** 2


def estimate_integral_monte_carlo(
    function: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    samples_count: int,
    random_seed: int = 42,
) -> float:
    """
    Обчислює наближення інтеграла ∫[lower_bound..upper_bound] function(x) dx методом Монте-Карло.

    Використовує геометричну інтерпретацію площі:
    - Створює прямокутник над відрізком [lower_bound, upper_bound] з висотою f_max.
    - Генерує випадкові точки та рахує, яка частина лежить під графіком.

    Параметри:
        function: функція f(x), яку інтегруємо
        lower_bound: нижня межа інтегрування
        upper_bound: верхня межа інтегрування
        samples_count: кількість випадкових точок (чим більше, тим точніше)
        random_seed: seed для відтворюваності результату

    Повертає:
        Оцінку інтеграла (float).
    """
    if samples_count <= 0:
        raise ValueError("samples_count має бути додатним цілим числом.")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound має бути більшим за lower_bound.")

    # Створюємо генератор випадкових чисел з фіксованим seed.
    random.seed(random_seed)

    # Рахуємо висоту прямокутника (для монотонної на відрізку функції цього достатньо).
    max_function_value = max(function(lower_bound), function(upper_bound))

    # Рахуємо площу прямокутника, у якому проводимо випадкове “влучання”.
    rectangle_area = (upper_bound - lower_bound) * max_function_value

    # Рахуємо кількість точок під графіком.
    points_under_curve = 0
    for _ in range(samples_count):
        x_random = random.uniform(lower_bound, upper_bound)
        y_random = random.uniform(0.0, max_function_value)

        if y_random <= function(x_random):
            points_under_curve += 1

    # Рахуємо оцінку інтеграла як частку “влучань” помножену на площу прямокутника.
    under_curve_ratio = points_under_curve / samples_count
    integral_estimate = under_curve_ratio * rectangle_area

    return integral_estimate


def analytic_integral_x_squared(lower_bound: float, upper_bound: float) -> float:
    """
    Обчислює аналітичне значення інтеграла ∫ x^2 dx на [lower_bound, upper_bound].

    Формула:
        ∫ x^2 dx = x^3 / 3
    """
    return (upper_bound ** 3 - lower_bound ** 3) / 3.0


def try_quad_integration(
    function: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
) -> Tuple[float, float] | None:
    """
    Перевіряє інтеграл через scipy.integrate.quad, якщо SciPy доступний.

    Повертає:
        (result, abs_error_estimate) якщо SciPy встановлено,
        або None, якщо SciPy відсутній.
    """
    try:
        import scipy.integrate as spi  # type: ignore
    except ImportError:
        return None

    result_value, abs_error_estimate = spi.quad(function, lower_bound, upper_bound)
    return float(result_value), float(abs_error_estimate)


def main() -> None:
    """Запускає демонстраційний розрахунок інтеграла методом Монте-Карло та перевірки."""
    # Створюємо межі інтегрування.
    lower_bound = 0.0
    upper_bound = 2.0

    # Створюємо параметри методу Монте-Карло.
    samples_count = 200_000

    # Рахуємо інтеграл методом Монте-Карло.
    monte_carlo_result = estimate_integral_monte_carlo(
        function=integrand_function,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        samples_count=samples_count,
        random_seed=42,
    )

    # Рахуємо аналітичний результат для порівняння.
    analytic_result = analytic_integral_x_squared(lower_bound, upper_bound)

    # Виводимо результати та різницю.
    print("Завдання 2: Інтеграл методом Монте-Карло")
    print(f"Монте-Карло (n={samples_count}): {monte_carlo_result}")
    print(f"Аналітичний результат: {analytic_result}")
    print(f"Абсолютна різниця: {abs(monte_carlo_result - analytic_result)}")

    # Перевіряємо через quad (опційно).
    quad_data = try_quad_integration(integrand_function, lower_bound, upper_bound)
    if quad_data is None:
        print("SciPy не встановлено. Для перевірки встановіть: pip install scipy")
        return

    quad_result, quad_error_estimate = quad_data
    print("Перевірка через scipy.integrate.quad")
    print(f"quad: {quad_result}, оцінка похибки: {quad_error_estimate}")
    print(f"|MC - quad|: {abs(monte_carlo_result - quad_result)}")


if __name__ == "__main__":
    main()
