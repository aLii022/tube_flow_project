import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def track_particles(x_grid, y_grid, z_grid, Ux, Uy, Uz, inside,
                    initial_positions, t_span, dt):
    """
    计算粒子轨迹。使用欧拉法积分。

    参数：
        x_grid, y_grid, z_grid: 一维坐标数组，单位 m
        Ux, Uy, Uz: 三维速度场，单位 m/s
        inside: 管内布尔掩膜
        initial_positions: 初始粒子位置，形状为 (N, 3)，单位 m
        t_span: 时间范围，例如 (0, 10)，单位 s
        dt: 时间步长，单位 s

    返回：
        trajectories: 每个粒子的轨迹点列表，每个元素为 (times, positions) 元组
    """
    Ux_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), Ux,
                                         bounds_error=False, fill_value=0.0)
    Uy_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), Uy,
                                         bounds_error=False, fill_value=0.0)
    Uz_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), Uz,
                                         bounds_error=False, fill_value=0.0)

    R = x_grid[-1]
    L = z_grid[-1]

    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt)) + 1
    times = np.linspace(t_start, t_end, n_steps)

    trajectories = []

    for i in range(initial_positions.shape[0]):
        pos = initial_positions[i].copy().astype(np.float64)
        positions = np.zeros((n_steps, 3))
        positions[0] = pos

        for step in range(1, n_steps):
            vx = float(Ux_interp(pos.reshape(1, 3))[0])
            vy = float(Uy_interp(pos.reshape(1, 3))[0])
            vz = float(Uz_interp(pos.reshape(1, 3))[0])
            pos = pos + np.array([vx, vy, vz]) * dt

            r_now = np.sqrt(pos[0]**2 + pos[1]**2)
            if r_now > R:
                radial_dir = np.array([pos[0], pos[1], 0.0])
                radial_dir_norm = np.linalg.norm(radial_dir)
                if radial_dir_norm > 1e-15:
                    radial_dir = radial_dir / radial_dir_norm
                pos[0] = R * 0.999 * radial_dir[0]
                pos[1] = R * 0.999 * radial_dir[1]

            if pos[2] < 0:
                pos[2] = 0.0
            if pos[2] > L:
                pos[2] = L

            positions[step] = pos

            if pos[2] >= L - 1e-8:
                positions = positions[:step + 1]
                break

        traj_times = times[:positions.shape[0]]
        trajectories.append((traj_times, positions))

    return trajectories


def visualize_particle_trajectories(trajectories, R, L, flow_type,
                                    output_dir="output"):
    """
    可视化粒子轨迹，保存为 PNG。

    参数：
        trajectories: track_particles 返回的轨迹列表
        R: 试管半径，单位 m
        L: 试管长度，单位 m
        flow_type: 流场类型名称
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    R_mm = R * 1e3
    L_mm = L * 1e3

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for i, (t_arr, pos) in enumerate(trajectories):
        pos_mm = pos * 1e3
        ax.plot(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
                color=colors[i], linewidth=1.0, alpha=0.8,
                label=f"Particle {i}")
        ax.scatter(*pos_mm[0], color=colors[i], s=25, marker="o",
                   edgecolors="black", linewidths=0.3, zorder=5)
        ax.scatter(*pos_mm[-1], color=colors[i], s=40, marker="s",
                   edgecolors="black", linewidths=0.3, zorder=5)

    theta_cyl = np.linspace(0, 2 * np.pi, 100)
    z_vals = np.array([0, L_mm])
    for zv in z_vals:
        ax.plot(R_mm * np.cos(theta_cyl), R_mm * np.sin(theta_cyl), zv,
                "gray", linewidth=0.5, alpha=0.3)

    z_lines = np.linspace(0, L_mm, 30)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([R_mm * np.cos(angle)] * len(z_lines),
                [R_mm * np.sin(angle)] * len(z_lines),
                z_lines, "gray", linewidth=0.3, alpha=0.15)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Particle Trajectories - {flow_type}")
    ax.set_xlim(-R_mm * 1.2, R_mm * 1.2)
    ax.set_ylim(-R_mm * 1.2, R_mm * 1.2)
    ax.set_zlim(0, L_mm)
    ax.set_box_aspect([1, 1, L_mm / R_mm])

    if len(trajectories) <= 10:
        ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{flow_type}_particle_trajectories.png"),
                dpi=150)
    plt.close(fig)

    print(f"[INFO] 粒子轨迹图已保存至 {output_dir}/")
