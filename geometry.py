import numpy as np


def create_tube_grid(R, L, nx, ny, nz):
    """
    创建圆柱试管所在的三维结构网格。

    参数：
        R: 试管半径，单位 m
        L: 试管长度，单位 m
        nx, ny, nz: x, y, z 方向的网格点数

    返回：
        X, Y, Z: 三维坐标数组，单位 m
        r: 半径数组，单位 m
        theta: 极角数组，单位 rad
        inside: 圆柱内部布尔掩膜
    """
    x = np.linspace(-R, R, nx)
    y = np.linspace(-R, R, ny)
    z = np.linspace(0, L, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    inside = r <= R

    return X, Y, Z, r, theta, inside
