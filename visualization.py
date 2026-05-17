import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def visualize_streamlines(X, Y, Z, Ux, Uy, Uz, speed, R, L, flow_type, output_dir="output"):
    """
    显示试管内三维流线。

    参数：
        X, Y, Z: 三维坐标数组
        Ux, Uy, Uz: 三维速度分量
        speed: 速度大小
        R: 试管半径
        L: 试管长度
        flow_type: 流场类型名称
        output_dir: 输出目录
    """
    if not HAS_PYVISTA:
        print("[WARNING] PyVista 未安装，跳过三维流线可视化。"
              " 请运行 pip install pyvista 安装。")
        return

    grid = pv.StructuredGrid(X, Y, Z)
    grid["Ux"] = Ux.ravel(order="F")
    grid["Uy"] = Uy.ravel(order="F")
    grid["Uz"] = Uz.ravel(order="F")
    grid["speed"] = speed.ravel(order="F")

    vec = np.column_stack((
        Ux.ravel(order="F"),
        Uy.ravel(order="F"),
        Uz.ravel(order="F"),
    ))
    grid["vectors"] = vec
    grid.set_active_vectors("vectors")

    n_seeds_radial = 4
    n_seeds_circ = 8
    angles = np.linspace(0, 2 * np.pi, n_seeds_circ, endpoint=False)
    radii = np.linspace(0.1 * R, 0.85 * R, n_seeds_radial)
    seed_points = []
    z_seed = 0.02 * L
    for angle in angles:
        for rad in radii:
            sx = rad * np.cos(angle)
            sy = rad * np.sin(angle)
            seed_points.append([sx, sy, z_seed])
    seed_points = np.array(seed_points)

    source = pv.PolyData(seed_points)
    streamlines = grid.streamlines_from_source(
        source,
        integrator_type=45,
        integration_direction="forward",
        initial_step_length=0.0005,
        max_step_length=0.01,
        max_length=20,
        max_steps=50000,
        progress_bar=False,
    )

    if streamlines.n_points > 0:
        ux = streamlines["Ux"]
        uy = streamlines["Uy"]
        uz = streamlines["Uz"]
        streamlines["speed"] = np.sqrt(ux**2 + uy**2 + uz**2)
        print(f"      已生成 {streamlines.n_points} 个流线点")
    else:
        print("      [WARNING] 未生成任何流线，请检查种子点位置和速度场")

    n_theta = 80
    n_z = 2
    theta_cyl = np.linspace(0, 2 * np.pi, n_theta)
    z_cyl = np.linspace(0, L, n_z)
    Theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_cyl = R * np.cos(Theta_cyl)
    Y_cyl = R * np.sin(Theta_cyl)
    xyz_cyl = np.zeros((n_z * n_theta, 3))
    xyz_cyl[:, 0] = X_cyl.ravel()
    xyz_cyl[:, 1] = Y_cyl.ravel()
    xyz_cyl[:, 2] = Z_cyl.ravel()
    cylinder = pv.StructuredGrid()
    cylinder.points = xyz_cyl
    cylinder.dimensions = (n_theta, n_z, 1)

    p = pv.Plotter(window_size=[1000, 700])
    p.add_mesh(cylinder, color="lightblue", opacity=0.15, show_edges=False)

    if streamlines.n_points > 0:
        p.add_mesh(streamlines, scalars="speed",
                   cmap="jet", render_lines_as_tubes=True,
                   line_width=3, scalar_bar_args={"title": "Speed (m/s)",
                                                   "vertical": True,
                                                   "height": 0.5,
                                                   "width": 0.08,
                                                   "position_x": 0.88,
                                                   "position_y": 0.25,
                                                   "label_font_size": 14,
                                                   "title_font_size": 16})
    p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    p.camera_position = "xz"
    p.camera.azimuth = 45
    p.camera.elevation = 20
    p.show(title=f"Tube Flow - {flow_type}", auto_close=False)


def plot_2d_slices(X, Y, Z, Ux, Uy, Uz, speed, R, L, flow_type, output_dir="output"):
    """
    绘制二维截面速度场。

    参数：
        X, Y, Z: 三维坐标数组
        Ux, Uy, Uz: 三维速度分量
        speed: 速度大小
        R: 试管半径
        L: 试管长度
        flow_type: 流场类型名称
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    R_mm = R * 1e3
    L_mm = L * 1e3
    X_mm = X * 1e3
    Y_mm = Y * 1e3
    Z_mm = Z * 1e3

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    iy_mid = Y.shape[1] // 2
    speed_y0 = speed[:, iy_mid, :]
    X_slice_y0 = X_mm[:, iy_mid, :]
    Z_slice_y0 = Z_mm[:, iy_mid, :]

    r_slice = np.sqrt(X_slice_y0**2 + Y_mm[:, iy_mid, :]**2)
    speed_y0_masked = np.where(r_slice <= R_mm, speed_y0, np.nan)

    c1 = ax1.pcolormesh(X_slice_y0, Z_slice_y0, speed_y0_masked * 1e3,
                         shading="auto", cmap="jet")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Z (mm)")
    ax1.set_title(f"速度大小云图 (y=0 纵截面) - {flow_type}")
    ax1.set_aspect("equal")
    ax1.set_xlim(-R_mm * 1.1, R_mm * 1.1)
    ax1.set_ylim(0, L_mm)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    fig1.colorbar(c1, ax=ax1, label="Speed (mm/s)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{flow_type}_longitudinal_slice.png"), dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 7))

    iz_mid = Z.shape[2] // 2
    speed_zmid = speed[:, :, iz_mid]
    Ux_zmid = Ux[:, :, iz_mid]
    Uy_zmid = Uy[:, :, iz_mid]
    X_slice_zmid = X_mm[:, :, iz_mid]
    Y_slice_zmid = Y_mm[:, :, iz_mid]

    r_cross = np.sqrt(X_slice_zmid**2 + Y_slice_zmid**2)
    speed_zmid_masked = np.where(r_cross <= R_mm, speed_zmid, np.nan)

    skip = max(1, X.shape[0] // 15)
    X_quiv = X_slice_zmid[::skip, ::skip]
    Y_quiv = Y_slice_zmid[::skip, ::skip]
    Ux_quiv = Ux_zmid[::skip, ::skip] * 1e3
    Uy_quiv = Uy_zmid[::skip, ::skip] * 1e3

    c2 = ax2.pcolormesh(X_slice_zmid, Y_slice_zmid, speed_zmid_masked * 1e3,
                         shading="auto", cmap="jet")
    ax2.quiver(X_quiv, Y_quiv, Ux_quiv, Uy_quiv, color="white",
               alpha=0.8, scale=5.0, width=0.003)
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(R_mm * np.cos(theta_circle), R_mm * np.sin(theta_circle),
             "w--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title(f"速度矢量图 (z=L/2 横截面) - {flow_type}")
    ax2.set_aspect("equal")
    ax2.set_xlim(-R_mm * 1.1, R_mm * 1.1)
    ax2.set_ylim(-R_mm * 1.1, R_mm * 1.1)
    fig2.colorbar(c2, ax=ax2, label="Speed (mm/s)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{flow_type}_cross_section.png"), dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    speed_inside = speed[X**2 + Y**2 <= R**2]
    ax3.hist(speed_inside * 1e3, bins=60, color="steelblue", edgecolor="white",
             alpha=0.85)
    ax3.set_xlabel("Speed (mm/s)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"速度大小分布 - {flow_type}")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, f"{flow_type}_speed_distribution.png"), dpi=150)
    plt.close(fig3)

    print(f"[INFO] 二维切片图已保存至 {output_dir}/")
