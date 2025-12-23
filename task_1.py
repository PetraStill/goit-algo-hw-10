"""
Модуль розв’язує задачу лінійного програмування для оптимізації виробництва напоїв.

Мета:
- Максимізуємо загальну кількість вироблених продуктів: "Лимонад" + "Фруктовий сік".

Обмеження:
- Враховуємо доступні ресурси (вода, цукор, лимонний сік, фруктове пюре).
- Враховуємо витрати ресурсів на одиницю кожного продукту.
- Змінні виробництва є невід’ємними цілими числами.

Результат:
- Повертаємо оптимальні обсяги виробництва та статус розв’язку.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pulp


@dataclass(frozen=True)
class AvailableResources:
    """Описує доступні обсяги сировини для виробництва."""
    water_units: int = 100
    sugar_units: int = 50
    lemon_juice_units: int = 30
    fruit_puree_units: int = 40


def solve_production_plan(resources: AvailableResources) -> Tuple[int, int, int, str]:
    """
    Розв’язує задачу оптимізації виробництва з використанням PuLP.

    Максимізує сумарну кількість:
        lemonade_units + fruit_juice_units

    За обмежень:
        2*lemonade_units + 1*fruit_juice_units <= water_units
        1*lemonade_units <= sugar_units
        1*lemonade_units <= lemon_juice_units
        2*fruit_juice_units <= fruit_puree_units

    Повертає:
        lemonade_units: оптимальна кількість "Лимонаду"
        fruit_juice_units: оптимальна кількість "Фруктового соку"
        total_units: сумарна кількість продуктів
        solve_status: текстовий статус розв’язку (наприклад, "Optimal")
    """
    # Створюємо модель ЛП з ціллю максимізації.
    model = pulp.LpProblem("Beverage_Production_Maximization", pulp.LpMaximize)

    # Створюємо змінні рішення (цілі, невід’ємні).
    lemonade_units = pulp.LpVariable("LemonadeUnits", lowBound=0, cat="Integer")
    fruit_juice_units = pulp.LpVariable("FruitJuiceUnits", lowBound=0, cat="Integer")

    # Максимізуємо загальну кількість вироблених одиниць.
    model += lemonade_units + fruit_juice_units, "Maximize_Total_Units"

    # Додаємо обмеження на воду.
    model += (
        2 * lemonade_units + 1 * fruit_juice_units <= resources.water_units
    ), "Water_Limit"

    # Додаємо обмеження на цукор (потрібен лише для лимонаду).
    model += (1 * lemonade_units <= resources.sugar_units), "Sugar_Limit"

    # Додаємо обмеження на лимонний сік (потрібен лише для лимонаду).
    model += (
        1 * lemonade_units <= resources.lemon_juice_units
    ), "LemonJuice_Limit"

    # Додаємо обмеження на фруктове пюре (потрібне лише для соку).
    model += (
        2 * fruit_juice_units <= resources.fruit_puree_units
    ), "FruitPuree_Limit"

    # Рахуємо оптимальний план (використовуємо CBC-солвер без зайвого виводу).
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Рахуємо значення змінних та статус.
    solve_status = pulp.LpStatus[model.status]
    lemonade_result = int(pulp.value(lemonade_units) or 0)
    fruit_juice_result = int(pulp.value(fruit_juice_units) or 0)
    total_result = lemonade_result + fruit_juice_result

    return lemonade_result, fruit_juice_result, total_result, solve_status


def main() -> None:
    """Запускає демонстраційний розрахунок оптимального плану виробництва."""
    # Створюємо доступні ресурси.
    resources = AvailableResources()

    # Рахуємо оптимальний план.
    lemonade_units, fruit_juice_units, total_units, solve_status = solve_production_plan(resources)

    # Виводимо результат.
    print("Завдання 1: Оптимізація виробництва (PuLP)")
    print(f"Статус: {solve_status}")
    print(f"Лимонад: {lemonade_units}")
    print(f"Фруктовий сік: {fruit_juice_units}")
    print(f"Загалом: {total_units}")


if __name__ == "__main__":
    main()
