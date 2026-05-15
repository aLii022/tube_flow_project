import os
import json
import numpy as np
import csv

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False



def export_velocity_field_vtk(X, Y, Z, Ux, Uy, Uz, speed, flow_type,
                              output_dir="output"):
    """
    导出速度场为 .vtk 文件。

    参数：
        X, Y, Z: 三维坐标数组
        Ux, Uy, Uz: 三维速度分量
        speed: 速度大小
        flow_type: 流场类型名称
        output_dir: 输出目录
    """
    if not HAS_PYVISTA:
        print("[WARNING] PyVista 未安装，跳过 VTK 导出。")
        return None

    os.makedirs(output_dir, exist_ok=True)

    grid = pv.StructuredGrid(X, Y, Z)
    grid["Ux"] = Ux.ravel(order="F")
    grid["Uy"] = Uy.ravel(order="F")
    grid["Uz"] = Uz.ravel(order="F")
    grid["speed"] = speed.ravel(order="F")

    filename = os.path.join(output_dir, f"{flow_type}_velocity.vtk")
    grid.save(filename)
    print(f"[INFO] 速度场已导出至 {filename}")
    return filename



def export_particle_trajectories_csv(trajectories, flow_type,
                                     output_dir="output"):
    """
    导出粒子轨迹为 .csv 文件。

    参数：
        trajectories: track_particles 返回的轨迹列表
        flow_type: 流场类型名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{flow_type}_trajectories.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "time_s", "x_m", "y_m", "z_m", "speed_m_s"])
        for pid, (t_arr, pos) in enumerate(trajectories):
            for step in range(len(t_arr)):
                t_val = t_arr[step]
                px, py, pz = pos[step]
                if step > 0:
                    dp = pos[step] - pos[step - 1]
                    spd = np.linalg.norm(dp) / (t_arr[step] - t_arr[step - 1]) \
                        if (t_arr[step] - t_arr[step - 1]) > 1e-12 else 0.0
                else:
                    spd = 0.0
                writer.writerow([pid, t_val, px, py, pz, spd])

    print(f"[INFO] 粒子轨迹已导出至 {filename}")
    return filename



def export_params_json(flow_type, params, output_dir="output"):
    """
    保存主要参数为 .json 文件。

    参数：
        flow_type: 流场类型名称
        params: 参数字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{flow_type}_params.json")

    output = {
        "flow_type": flow_type,
        "params": {k: (float(v) if isinstance(v, (np.floating, np.integer))
                        else v) for k, v in params.items()}
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 参数已导出至 {filename}")
    return filename
