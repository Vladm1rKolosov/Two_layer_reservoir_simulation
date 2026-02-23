import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv, kn, factorial
from typing import Dict, List, Tuple, Callable, Union, Any, Optional


class TwoLayerReservoir:
    """Модель двухслойного пласта для гидродинамических исследований скважин.

    Класс реализует математическую модель фильтрации флюида в двух пластах
    с различными фильтрационно-емкостными свойствами. Расчет нестационарного процесса фильтрации
    реализуется аналитически при помощи преобразования Лапласа.

    Params:
        p_initial (float): Начальное пластовое давление, Па
        rw (float): Радиус скважины, м
        mu_cp (float): Вязкость в сантипуазах, сП
        mu (float): Вязкость в Па·с
        Q_m3_day (float): Дебит скважины, м³/сут
        Q_total (float): Дебит скважины, м³/с
        c_atm (float): Сжимаемость в 1/атм
        c (float): Сжимаемость в 1/Па
        re1 (float): Радиус контура питания первого пласта, м
        k1 (float): Проницаемость первого пласта, м²
        m1 (float): Пористость первого пласта
        h1 (float): Мощность первого пласта, м
        S1 (float): Скин-фактор первого пласта
        re2 (float): Радиус контура питания второго пласта, м
        k2 (float): Проницаемость второго пласта, м²
        m2 (float): Пористость второго пласта
        h2 (float): Мощность второго пласта, м
        S2 (float): Скин-фактор второго пласта
        kappa1 (float): Пьезопроводность первого пласта, м²/с
        kappa2 (float): Пьезопроводность второго пласта, м²/с
        kpr1 (float): Гидропроводность первого пласта, м³·с/кг
        kpr2 (float): Гидропроводность второго пласта, м³·с/кг
    """

    def __init__(self: "TwoLayerReservoir") -> None:
        """Инициализация модели с параметрами по умолчанию."""
        # Параметры
        self.p_initial: float = 250 * 101325  # Начальное давление, Па
        self.rw: float = 0.1  # Радиус скважины, м
        self.mu_cp: float = 1.5  # Вязкость, сП
        self.mu: float = self.mu_cp * 1e-3  # Вязкость, Па·с
        self.Q_m3_day: float = 100  # Дебит, м³/сут
        self.Q_total: float = self.Q_m3_day / 86400  # Дебит, м³/с

        # Сжимаемость
        self.c_atm: float = 5e-5  # 1/атм
        self.c: float = self.c_atm / 101325  # 1/Па

        # Пласт 1
        self.re1: float = 250  # Контур питания, м
        self.k1: float = 10e-15  # Проницаемость, м²
        self.m1: float = 0.2  # Пористость
        self.h1: float = 10  # Мощность, м
        self.S1: float = 0  # Скин-фактор

        # Пласт 2
        self.re2: float = 50  # Контур питания, м
        self.k2: float = 10e-15  # Проницаемость, м²
        self.m2: float = 0.2  # Пористость
        self.h2: float = 10  # Мощность, м
        self.S2: float = 0  # Скин-фактор

        # Пьезопроводности
        self.kappa1: float = self.k1 / (self.mu * self.c * self.m1)
        self.kappa2: float = self.k2 / (self.mu * self.c * self.m2)

        # Проводимости
        self.kpr1: float = 2 * np.pi * self.k1 * self.h1 / self.mu
        self.kpr2: float = 2 * np.pi * self.k2 * self.h2 / self.mu

    def stehfest_invert(
        self: "TwoLayerReservoir", F: Callable[[float], complex], t: float, n: int = 8
    ) -> float:
        """Обращение преобразования Лапласа методом Стефеста.

        Реализует численное обращение преобразования Лапласа с использованием
        алгоритма Стефеста.

        Params:
            F: Функция-изображение в пространстве Лапласа, принимает параметр s
                и возвращает комплексное значение изображения
            t: Время, для которого вычисляется оригинал, с
            n: Параметр метода, определяет точность.

        Returns:
            float: Значение оригинала функции во временной области в момент t
        """

        if n % 2 != 0:
            n += 1

        ln2 = np.log(2.0)
        v = np.zeros(n + 1)

        # Коэффициенты Стефеста
        for i in range(1, n + 1):
            s = 0.0
            for k in range((i + 1) // 2, min(i, n // 2) + 1):
                s += (k ** (n // 2) * factorial(2 * k)) / (
                    factorial(n // 2 - k)
                    * factorial(k)
                    * factorial(k - 1)
                    * factorial(i - k)
                    * factorial(2 * k - i)
                )
            v[i] = s * (-1) ** (i + n // 2)

        # Обратное преобразование
        result = 0.0
        for i in range(1, n + 1):
            s_val = i * ln2 / t
            result += v[i] * F(s_val) * ln2 / t

        return result

    def get_bessel(
        self: "TwoLayerReservoir", s: float, layer_num: int, r: Optional[float] = None
    ) -> Dict[str, Union[float, complex]]:
        """Вычисление модифицированных функций Бесселя для заданного пласта.

        Рассчитывает значения модифицированных функций Бесселя первого и второго
        рода нулевого и первого порядка.

        Params:
            s: Параметр преобразования Лапласа, 1/с
            layer_num: Номер пласта (1 или 2)
            r: Радиус точки, для которой вычисляются функции, м.
                Если None, используется радиус скважины

        Returns:
            Dict[str, Union[float, complex]]: Словарь, содержащий:
                - z: аргумент функций Бесселя
                - re: радиус контура питания
                - I0_re, K0_re: функции при r = re
                - I1_re, K1_re: функции первого порядка при r = re
                - I0_rw, K0_rw: функции при r = rw
                - I1_rw, K1_rw: функции первого порядка при r = rw
                - I0_r, K0_r: функции при заданном r
                - I1_r, K1_r: функции первого порядка при заданном r
        """
        if layer_num == 1:
            z = np.sqrt(s / self.kappa1)
            re = self.re1
        else:
            z = np.sqrt(s / self.kappa2)
            re = self.re2

        r_val = r if r is not None else self.rw

        # Функции Бесселя на границах
        I0_re = iv(0, z * re)
        K0_re = kn(0, z * re)
        I1_re = iv(1, z * re)
        K1_re = kn(1, z * re)

        I0_rw = iv(0, z * self.rw)
        K0_rw = kn(0, z * self.rw)
        I1_rw = iv(1, z * self.rw)
        K1_rw = kn(1, z * self.rw)

        # Функции в точке r (если задана)
        if r is not None and r != self.rw:
            I0_r = iv(0, z * r)
            K0_r = kn(0, z * r)
            I1_r = iv(1, z * r)
            K1_r = kn(1, z * r)
        else:
            I0_r, K0_r, I1_r, K1_r = I0_rw, K0_rw, I1_rw, K1_rw

        return {
            "z": z,
            "re": re,
            "I0_re": I0_re,
            "K0_re": K0_re,
            "I1_re": I1_re,
            "K1_re": K1_re,
            "I0_rw": I0_rw,
            "K0_rw": K0_rw,
            "I1_rw": I1_rw,
            "K1_rw": K1_rw,
            "I0_r": I0_r,
            "K0_r": K0_r,
            "I1_r": I1_r,
            "K1_r": K1_r,
        }

    def get_coefficients(
        self: "TwoLayerReservoir", s: float
    ) -> Tuple[complex, complex]:
        """Коэффициенты C11 и C21 из системы уравнений.

        Решает систему линейных уравнений для определения коэффициентов
        разложения в каждом пласте.

        Params:
            s: Параметр преобразования Лапласа, 1/с

        Returns:
            Tuple[complex, complex]: Кортеж (C11, C21) - коэффициенты для первого и второго пластов
        """
        if abs(s) < 1e-30:
            return 0j, 0j

        b1 = self.get_bessel(s, 1)
        b2 = self.get_bessel(s, 2)

        # Коэффициенты системы
        A11 = (
            b1["I0_rw"]
            - (b1["I0_re"] / b1["K0_re"]) * b1["K0_rw"]
            - self.S1
            * self.rw
            * b1["z"]
            * (b1["I1_rw"] + (b1["I0_re"] / b1["K0_re"]) * b1["K1_rw"])
        )

        A12 = -(
            b2["I0_rw"]
            + (b2["I1_re"] / b2["K1_re"]) * b2["K0_rw"]
            - self.S2
            * self.rw
            * b2["z"]
            * (b2["I1_rw"] - (b2["I1_re"] / b2["K1_re"]) * b2["K1_rw"])
        )

        B11 = (
            self.rw
            * self.kpr1
            * b1["z"]
            * (b1["I1_rw"] + (b1["I0_re"] / b1["K0_re"]) * b1["K1_rw"])
        )
        B12 = (
            self.rw
            * self.kpr2
            * b2["z"]
            * (b2["I1_rw"] - (b2["I1_re"] / b2["K1_re"]) * b2["K1_rw"])
        )

        A = np.array([[A11, A12], [B11, B12]], dtype=complex)
        b = np.array([0, self.Q_total / s], dtype=complex)

        try:
            C11, C21 = np.linalg.solve(A, b)
            return C11, C21
        except np.linalg.LinAlgError:
            return 0j, 0j

    def pressure_disturbance_at_well(self: "TwoLayerReservoir", s: float) -> float:
        """Возмущение давления на забое.

        Вычисляет изменение давления на забое скважины относительно начального
        в пространстве Лапласа.

        Params:
            s: Параметр преобразования Лапласа, 1/с

        Returns:
            float: Возмущение давления на забое, Па
        """
        if abs(s) < 1e-30:
            # Стационарное решение
            R1 = (np.log(self.re1 / self.rw) + self.S1) / self.kpr1
            R2 = (np.log(self.re2 / self.rw) + self.S2) / self.kpr2
            R_total = 1.0 / (1.0 / R1 + 1.0 / R2)
            return -self.Q_total * R_total

        C11, C21 = self.get_coefficients(s)
        b1 = self.get_bessel(s, 1)
        b2 = self.get_bessel(s, 2)

        # Давление без скина
        p1 = C11 * b1["I0_rw"] - C11 * (b1["I0_re"] / b1["K0_re"]) * b1["K0_rw"]
        p2 = C21 * b2["I0_rw"] + C21 * (b2["I1_re"] / b2["K1_re"]) * b2["K0_rw"]

        # Дебиты
        q1 = (
            self.rw
            * self.kpr1
            * b1["z"]
            * (C11 * b1["I1_rw"] + C11 * (b1["I0_re"] / b1["K0_re"]) * b1["K1_rw"])
        )
        q2 = (
            self.rw
            * self.kpr2
            * b2["z"]
            * (C21 * b2["I1_rw"] - C21 * (b2["I1_re"] / b2["K1_re"]) * b2["K1_rw"])
        )

        # Падение давления из-за скина
        delta_skin1 = (self.mu * self.S1 / (2 * np.pi * self.k1 * self.h1)) * q1
        delta_skin2 = (self.mu * self.S2 / (2 * np.pi * self.k2 * self.h2)) * q2

        # Возмущение давления на забое
        p_disturbance1 = p1 - delta_skin1
        p_disturbance2 = p2 - delta_skin2

        return float(np.real((p_disturbance1 + p_disturbance2) / 2))

    def flow_rate(self: "TwoLayerReservoir", layer_num: int, s: float) -> float:
        """Дебит из пласта.

        Вычисляет дебит флюида из указанного пласта в пространстве Лапласа.

        Params:
            layer_num: Номер пласта
            s: Параметр преобразования Лапласа, 1/с

        Returns:
            float: Дебит из пласта, м³/с
        """
        if abs(s) < 1e-30:
            # Стационарное решение
            R1 = (np.log(self.re1 / self.rw) + self.S1) / self.kpr1
            R2 = (np.log(self.re2 / self.rw) + self.S2) / self.kpr2

            if layer_num == 1:
                return self.Q_total * R2 / (R1 + R2)
            else:
                return self.Q_total * R1 / (R1 + R2)

        C11, C21 = self.get_coefficients(s)
        b = self.get_bessel(s, layer_num)

        if layer_num == 1:
            C12 = -C11 * b["I0_re"] / b["K0_re"]
            rate = self.rw * self.kpr1 * b["z"] * (C11 * b["I1_rw"] - C12 * b["K1_rw"])
        else:
            C22 = C21 * b["I1_re"] / b["K1_re"]
            rate = self.rw * self.kpr2 * b["z"] * (C21 * b["I1_rw"] - C22 * b["K1_rw"])

        return float(np.real(rate))

    def pressure_disturbance_at_radius(
        self: "TwoLayerReservoir", layer_num: int, r: float, s: float
    ) -> float:
        """Возмущение давления в точке r.

        Вычисляет изменение давления на заданном расстоянии от скважины
        в указанном пласте в пространстве Лапласа.

        Params:
            layer_num: Номер пласта
            r: Радиус точки, м
            s: Параметр преобразования Лапласа, 1/с

        Returns:
            float: Возмущение давления в точке r, Па
        """
        if abs(s) < 1e-30:
            # Стационарное решение
            R1 = (np.log(self.re1 / self.rw) + self.S1) / self.kpr1
            R2 = (np.log(self.re2 / self.rw) + self.S2) / self.kpr2

            if layer_num == 1:
                q = self.Q_total * R2 / (R1 + R2)
                return -q * np.log(self.re1 / r) / self.kpr1
            else:
                q = self.Q_total * R1 / (R1 + R2)
                return -q * np.log(self.re2 / r) / self.kpr2

        C11, C21 = self.get_coefficients(s)
        b = self.get_bessel(s, layer_num, r)

        if layer_num == 1:
            C12 = -C11 * b["I0_re"] / b["K0_re"]
            disturbance = C11 * b["I0_r"] + C12 * b["K0_r"]
        else:
            C22 = C21 * b["I1_re"] / b["K1_re"]
            disturbance = C21 * b["I0_r"] + C22 * b["K0_r"]

        return float(np.real(disturbance))


def calculate_time_data(
    reservoir: TwoLayerReservoir, time_hours: np.ndarray
) -> Dict[str, np.ndarray]:
    """Расчет зависимости параметров от времени.

    Вычисляет забойное давление, падение давления и дебиты по пластам
    в зависимости от времени путем численного обращения преобразования Лапласа.

    Params:
        reservoir: Модель двухслойного пласта
        time_hours: Массив времени в часах

    Returns:
        Dict[str, np.ndarray]: Словарь с полями:
            - time: массив времени, ч
            - p_well: забойное давление, Па
            - delta_p: падение давления, Па
            - q1: дебит первого пласта, м³/с
            - q2: дебит второго пласта, м³/с
            - q_total: суммарный дебит, м³/с
    """
    time_sec = time_hours * 3600

    p_well, delta_p = [], []
    q1, q2 = [], []

    for t in time_sec:
        # Возмущение давления
        def laplace_func(s: float) -> float:
            return reservoir.pressure_disturbance_at_well(s)

        disturbance = reservoir.stehfest_invert(laplace_func, t, n=16)

        # Падение давления
        delta_p_t = -disturbance
        delta_p.append(delta_p_t)

        # Забойное давление
        p_well_t = reservoir.p_initial + disturbance
        p_well.append(p_well_t)

        # Дебиты
        def q1_func(s: float) -> float:
            return reservoir.flow_rate(1, s)

        def q2_func(s: float) -> float:
            return reservoir.flow_rate(2, s)

        q1.append(reservoir.stehfest_invert(q1_func, t, n=16))
        q2.append(reservoir.stehfest_invert(q2_func, t, n=16))

    q_total = [q1[i] + q2[i] for i in range(len(q1))]

    return {
        "time": time_hours,
        "p_well": np.array(p_well),
        "delta_p": np.array(delta_p),
        "q1": np.array(q1),
        "q2": np.array(q2),
        "q_total": np.array(q_total),
    }


def calculate_radial_profiles(
    reservoir: TwoLayerReservoir, times_hours: List[float]
) -> Tuple[np.ndarray, np.ndarray, List[float], List[List[float]], List[List[float]]]:
    """Расчет распределения давления по радиусу для заданных моментов времени.

    Вычисляет профили давления в каждом пласте для указанных времен.

    Params:
        reservoir: Модель двухслойного пласта
        times_hours: Список времен в часах для расчета профилей

    Returns:
        Tuple[np.ndarray, np.ndarray, List[float], List[List[float]], List[List[float]]]:
            - r1: массив радиусов для первого пласта, м
            - r2: массив радиусов для второго пласта, м
            - times: список времен, ч
            - p1_profiles: список профилей давления для первого пласта
            - p2_profiles: список профилей давления для второго пласта
    """
    times_sec = [t * 3600 for t in times_hours]

    # Радиусы
    r1 = np.linspace(reservoir.rw, reservoir.re1, 100)
    r2 = np.linspace(reservoir.rw, reservoir.re2, 100)

    p1_profiles, p2_profiles = [], []

    for t in times_sec:
        p1_t, p2_t = [], []

        # Пласт 1
        for r in r1:

            def laplace_func(s: float) -> float:
                return reservoir.pressure_disturbance_at_radius(1, r, s)

            disturbance = reservoir.stehfest_invert(laplace_func, t, n=12)
            p1_t.append(reservoir.p_initial + disturbance)

        # Пласт 2
        for r in r2:

            def laplace_func(s: float) -> float:
                return reservoir.pressure_disturbance_at_radius(2, r, s)

            disturbance = reservoir.stehfest_invert(laplace_func, t, n=12)
            p2_t.append(reservoir.p_initial + disturbance)

        p1_profiles.append(p1_t)
        p2_profiles.append(p2_t)

    return r1, r2, times_hours, p1_profiles, p2_profiles


def plot_results(
    reservoir: TwoLayerReservoir, time_data: Dict[str, np.ndarray]
) -> None:
    """Графики временных зависимостей + производная d(Δp)/dlog(t).

    Строит три графика:
    1. Забойное давление от времени в логарифмическом масштабе
    2. Дебиты по пластам от времени в логарифмическом масштабе
    3. Производную падения давления по логарифму времени

    Params:
        reservoir: Модель двухслойного пласта
        time_data: Данные временных зависимостей
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    # Убираем подписи осей X у верхних графиков
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    # 1) Забойное давление
    p_atm = time_data["p_well"] / 101325
    p_steady = np.mean(p_atm[-5:])
    p_init = reservoir.p_initial / 101325

    ax1.plot(
        time_data["time"],
        p_atm,
        "b-",
        linewidth=3,
        marker="o",
        markersize=6,
        markevery=10,
        label="Забойное давление",
    )

    ax1.axhline(
        p_init,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Начальное ({p_init:.0f} атм)",
        alpha=0.7,
    )
    ax1.axhline(
        p_steady,
        color="g",
        linestyle="-.",
        linewidth=2,
        label=f"Установившееся ({p_steady:.1f} атм)",
        alpha=0.7,
    )

    # Перемещаем подпись оси X вправо
    ax1.set_xlabel("Время, ч", fontsize=12, fontweight="bold", x=1.0, ha="right")
    ax1.set_ylabel("Давление, атм", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Забойное давление (S1={reservoir.S1}, S2={reservoir.S2})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_xscale("log")
    ax1.set_ylim([min(p_atm) * 0.95, p_init * 1.01])

    # 2) Дебиты
    q1_day = time_data["q1"] * 86400
    q2_day = time_data["q2"] * 86400
    q_total_day = time_data["q_total"] * 86400

    ax2.plot(
        time_data["time"],
        q1_day,
        "g-",
        linewidth=2.5,
        marker="^",
        markersize=6,
        markevery=10,
        label=f"Пласт 1 (S={reservoir.S1})",
    )

    ax2.plot(
        time_data["time"],
        q2_day,
        "orange",
        linewidth=2.5,
        marker="s",
        markersize=6,
        markevery=10,
        label=f"Пласт 2 (S={reservoir.S2})",
    )

    ax2.plot(
        time_data["time"],
        q_total_day,
        "b-",
        linewidth=3,
        marker="o",
        markersize=7,
        markevery=10,
        label="Суммарный",
    )

    ax2.axhline(
        reservoir.Q_m3_day,
        color="k",
        linestyle="--",
        linewidth=2,
        label=f"Заданный Q = {reservoir.Q_m3_day} м³/сут",
        alpha=0.7,
    )

    # Перемещаем подпись оси X вправо
    ax2.set_xlabel("Время, ч", fontsize=12, fontweight="bold", x=1.0, ha="right")
    ax2.set_ylabel("Дебит, м³/сут", fontsize=12, fontweight="bold")
    ax2.set_title("Дебиты от времени", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(fontsize=11, loc="upper left")
    ax2.set_xscale("log")
    ax2.set_ylim([0, max(q_total_day) * 1.2])

    # Аналитические стационарные дебиты
    R1 = (np.log(reservoir.re1 / reservoir.rw) + reservoir.S1) / reservoir.kpr1
    R2 = (np.log(reservoir.re2 / reservoir.rw) + reservoir.S2) / reservoir.kpr2
    q1_stat = reservoir.Q_total * R2 / (R1 + R2) * 86400
    q2_stat = reservoir.Q_total * R1 / (R1 + R2) * 86400

    t_h = np.asarray(time_data["time"], dtype=float)
    delta_p_atm = np.asarray(time_data["delta_p"], dtype=float) / 101325

    d_delta_p_dlogt = np.gradient(delta_p_atm, np.log10(t_h))

    ax3.plot(
        t_h,
        d_delta_p_dlogt,
        color="purple",
        linewidth=2.5,
        marker="d",
        markersize=5,
        markevery=12,
        label="d(Δp)/dlog10(t)",
    )

    # Перемещаем подпись оси X вправо
    ax3.set_xlabel("Время, ч", fontsize=12, fontweight="bold", x=1.0, ha="right")
    ax3.set_ylabel("d(Δp)/dlog10(t), атм", fontsize=12, fontweight="bold")
    ax3.set_title("Производная давления по log(t)", fontsize=14, fontweight="bold")
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.legend(fontsize=11, loc="upper left")
    ax3.set_xscale("log")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.2)
    plt.show()


def plot_radial_profiles(
    reservoir: TwoLayerReservoir,
    profile_data: Tuple[
        np.ndarray, np.ndarray, List[float], List[List[float]], List[List[float]]
    ],
) -> None:
    """Графики распределения давления по радиусу.

    Строит профили давления в каждом пласте для разных моментов времени.

    Args:
        reservoir: Модель двухслойного пласта
        profile_data: Кортеж с данными профилей из calculate_radial_profiles:
        (r1, r2, times, p1_profiles, p2_profiles)
    """
    r1, r2, times, p1_profiles, p2_profiles = profile_data

    # Конвертация в атм
    p_init = reservoir.p_initial / 101325
    p1_atm = [[p / 101325 for p in profile] for profile in p1_profiles]
    p2_atm = [[p / 101325 for p in profile] for profile in p2_profiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = ["red", "orange", "green", "blue", "purple"]
    markers = ["o", "s", "^", "v", "D"]

    # Пласт 1
    for i, (t, profile, color, marker) in enumerate(
        zip(times, p1_atm, colors, markers)
    ):
        ax1.plot(
            r1,
            profile,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markevery=10,
            label=f"t = {t} ч",
        )

    ax1.axhline(
        p_init,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Начальное ({p_init:.0f} атм)",
        alpha=0.7,
    )

    ax1.set_xlabel("Радиус, м", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Давление, атм", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Пласт 1 (S={reservoir.S1}, re={reservoir.re1} м)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xlim([0, reservoir.re1 * 1.05])

    # Пласт 2
    for i, (t, profile, color, marker) in enumerate(
        zip(times, p2_atm, colors, markers)
    ):
        ax2.plot(
            r2,
            profile,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markevery=10,
            label=f"t = {t} ч",
        )

    ax2.axhline(
        p_init,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Начальное ({p_init:.0f} атм)",
        alpha=0.7,
    )

    ax2.set_xlabel("Радиус, м", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Давление, атм", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Пласт 2 (S={reservoir.S2}, re={reservoir.re2} м)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim([0, reservoir.re2 * 1.05])

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def main() -> None:
    """Основная функция программы.

    Создает модель двухслойного пласта, выполняет расчет временных зависимостей
    и распределений давления по радиусу, строит соответствующие графики.
    """
    # Создание модели
    reservoir = TwoLayerReservoir()

    # Временные зависимости
    time_hours = np.logspace(-1, 3, 300)  # 0.1 ч до 1000 ч
    time_data = calculate_time_data(reservoir, time_hours)

    # Построение графиков
    plot_results(reservoir, time_data)

    # Распределение давления по радиусу
    selected_times = [0.1, 1, 10, 100, 1000]
    profile_data = calculate_radial_profiles(reservoir, selected_times)
    plot_radial_profiles(reservoir, profile_data)


if __name__ == "__main__":
    main()
