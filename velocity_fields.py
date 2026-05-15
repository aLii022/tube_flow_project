import numpy as np


def compute_velocity_field(X, Y, Z, R, L, flow_type, params):
    """
    根据 flow_type 计算自定义速度场。

    参数：
        X, Y, Z: 三维坐标数组，单位 m
        R: 试管半径，单位 m
        L: 试管长度，单位 m
        flow_type: 流场类型字符串
        params: 流场参数字典

    返回：
        Ux, Uy, Uz: 笛卡尔坐标系下的三维速度分量，单位 m/s
        speed: 速度大小，单位 m/s
    """
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    inside = r <= R

    if flow_type == "axial":
        Ur, Utheta, Uz = _compute_axial(r, inside, R, params)
    elif flow_type == "swirl":
        Ur, Utheta, Uz = _compute_swirl(r, inside, R, params)
    elif flow_type == "helical":
        Ur, Utheta, Uz = _compute_helical(r, inside, R, params)
    elif flow_type == "recirculation":
        Ur, Utheta, Uz = _compute_recirculation(r, Z, inside, R, L, params)
    elif flow_type == "acoustic_like":
        Ur, Utheta, Uz = _compute_acoustic_like(r, theta, Z, inside, R, L, params)
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    Ux = Ur * np.cos(theta) - Utheta * np.sin(theta)
    Uy = Ur * np.sin(theta) + Utheta * np.cos(theta)

    Ux = np.where(inside, Ux, 0.0)
    Uy = np.where(inside, Uy, 0.0)
    Uz = np.where(inside, Uz, 0.0)

    speed = np.sqrt(Ux**2 + Uy**2 + Uz**2)
    speed = np.where(inside, speed, 0.0)

    return Ux, Uy, Uz, speed


def _compute_axial(r, inside, R, params):
    """
    纯轴向层流流场。

    uz = Umax * (1 - r^2/R^2),  ur = 0,  utheta = 0
    """
    Umax = params.get("Umax", 2e-3)
    Ur = np.zeros_like(r)
    Utheta = np.zeros_like(r)
    Uz = np.where(inside, Umax * (1.0 - r**2 / R**2), 0.0)
    return Ur, Utheta, Uz


def _compute_swirl(r, inside, R, params):
    """
    纯旋涡流场。

    utheta = omega * r * (1 - r^2/R^2),  ur = 0,  uz = 0
    """
    omega = params.get("omega", 3.0)
    Ur = np.zeros_like(r)
    Utheta = np.where(inside, omega * r * (1.0 - r**2 / R**2), 0.0)
    Uz = np.zeros_like(r)
    return Ur, Utheta, Uz


def _compute_helical(r, inside, R, params):
    """
    轴向流 + 旋涡流的螺旋流场。

    uz = Umax * (1 - r^2/R^2)
    utheta = omega * r * (1 - r^2/R^2)
    """
    Umax = params.get("Umax", 2e-3)
    omega = params.get("omega", 3.0)
    Ur = np.zeros_like(r)
    Utheta = np.where(inside, omega * r * (1.0 - r**2 / R**2), 0.0)
    Uz = np.where(inside, Umax * (1.0 - r**2 / R**2), 0.0)
    return Ur, Utheta, Uz


def _compute_recirculation(r, Z, inside, R, L, params):
    """
    局部回流流场。

    ur = A * (r/R) * (1 - r/R) * sin(2*pi*z/L)
    uz = A * cos(pi*r/R) * cos(2*pi*z/L)
    utheta = 0
    """
    A = params.get("A", 0.8e-3)
    safe_r = np.where(r < R, r, R * 0.999)
    Ur = np.where(inside, A * (safe_r / R) * (1.0 - safe_r / R) * np.sin(2.0 * np.pi * Z / L), 0.0)
    Utheta = np.zeros_like(r)
    Uz = np.where(inside, A * np.cos(np.pi * safe_r / R) * np.cos(2.0 * np.pi * Z / L), 0.0)
    return Ur, Utheta, Uz


def _compute_acoustic_like(r, theta, Z, inside, R, L, params):
    """
    类超声声流的简化流场。

    ur = A * (r/R) * (1 - r/R) * sin(k*z)
    uz = A * cos(pi*r/R) * cos(k*z)
    utheta = B * r * (1 - r^2/R^2) * sin(k*z)

    其中 k = 2*pi*n / L，n 为轴向周期数。
    """
    A = params.get("A", 1.0e-3)
    B = params.get("B", 2.0)
    n_periods = params.get("n_periods", 2)
    k = 2.0 * np.pi * n_periods / L

    safe_r = np.where(r < R, r, R * 0.999)

    Ur = np.where(inside, A * (safe_r / R) * (1.0 - safe_r / R) * np.sin(k * Z), 0.0)
    Utheta = np.where(inside, B * safe_r * (1.0 - safe_r**2 / R**2) * np.sin(k * Z), 0.0)
    Uz = np.where(inside, A * np.cos(np.pi * safe_r / R) * np.cos(k * Z), 0.0)

    return Ur, Utheta, Uz
