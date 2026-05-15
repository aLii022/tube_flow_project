import os
import sys
import numpy as np

from config import (R, L, nx, ny, nz, flow_type,
                    Umax, omega, A, B, n_periods,
                    dt, t_max, n_particles, output_dir)
from geometry import create_tube_grid
from velocity_fields import compute_velocity_field
from visualization import visualize_streamlines, plot_2d_slices
from particle_tracking import (track_particles,
                               visualize_particle_trajectories)
from export_utils import (export_velocity_field_vtk,
                          export_particle_trajectories_csv,
                          export_params_json)


def main():
    print("=" * 60)
    print("  试管内自定义流场与流动状态可视化")
    print("=" * 60)
    print(f"\n当前流场类型: {flow_type}")

    print("\n[1/6] 创建试管几何网格 ...")
    X, Y, Z, r, theta, inside = create_tube_grid(R, L, nx, ny, nz)
    print(f"      网格尺寸: {X.shape}")

    params = {
        "Umax": Umax,
        "omega": omega,
        "A": A,
        "B": B,
        "n_periods": n_periods,
    }

    print("[2/6] 计算自定义速度场 ...")
    Ux, Uy, Uz, speed = compute_velocity_field(
        X, Y, Z, R, L, flow_type, params
    )
    print(f"      速度范围: {speed[speed > 0].min()*1e3:.3f} ~ "
          f"{speed.max()*1e3:.3f} mm/s")

    print("[3/6] 三维流线可视化 ...")
    visualize_streamlines(X, Y, Z, Ux, Uy, Uz, speed, R, L,
                          flow_type, output_dir)

    print("[4/6] 二维截面图 ...")
    plot_2d_slices(X, Y, Z, Ux, Uy, Uz, speed, R, L, flow_type, output_dir)

    print("[5/6] 粒子轨迹追踪 ...")
    x_1d = np.linspace(-R, R, nx)
    y_1d = np.linspace(-R, R, ny)
    z_1d = np.linspace(0, L, nz)

    angles_init = np.linspace(0, 2 * np.pi, n_particles, endpoint=False)
    r_init = 0.3 * R
    initial_positions = np.column_stack([
        r_init * np.cos(angles_init),
        r_init * np.sin(angles_init),
        np.full(n_particles, 0.01 * L),
    ])

    trajectories = track_particles(
        x_1d, y_1d, z_1d,
        Ux, Uy, Uz, inside,
        initial_positions, (0.0, t_max), dt
    )
    print(f"      已计算 {len(trajectories)} 条粒子轨迹")

    visualize_particle_trajectories(trajectories, R, L, flow_type, output_dir)

    print("[6/6] 结果导出 ...")
    export_velocity_field_vtk(X, Y, Z, Ux, Uy, Uz, speed, flow_type, output_dir)
    export_particle_trajectories_csv(trajectories, flow_type, output_dir)
    export_params_json(flow_type, params, output_dir)

    print("\n" + "=" * 60)
    print(f"  所有结果已保存至 '{output_dir}/' 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
