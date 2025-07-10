import torch
import numpy as np
import time
import os
from tqdm import tqdm

# ====== 1. 用户定义量子数范围 ======
quantum_ranges_per_group = [
    (6, 6, 2, 2),
    (6, 6, 2, 2),
    (6, 6, 2, 2)
]

# ====== 2. 生成所有状态组合 ======
def generate_all_states(ranges_per_group):
    ranges = []
    for (rn, rm, rk, rs) in ranges_per_group:
        ranges += [torch.arange(rn), torch.arange(rm), torch.arange(rk), torch.arange(rs)]
    grid = torch.meshgrid(*ranges, indexing='ij')
    all_states = torch.stack([g.reshape(-1) for g in grid], dim=1)
    return all_states  # (N, D)

# ====== 3. GPU 上的 H(q1, q2) 批量计算类 ======
class HValueCalculatorGPU:
    def __init__(self, connection_ratio=0.1):
        self.connection_ratio = connection_ratio

    def compute(self, q1, q2):
        diff = q1.unsqueeze(1) - q2.unsqueeze(0)
        hamming = diff.abs().sum(dim=2)
        hamming1_mask = (hamming == 1)

        if self.connection_ratio < 1.0:
            rand_mask = torch.rand_like(hamming1_mask.float()) < self.connection_ratio
            hamming1_mask = hamming1_mask & rand_mask.bool()

        H = torch.zeros((q1.size(0), q2.size(0)), device=q1.device)

        if q1.size(0) == q2.size(0):
            diag_indices = torch.arange(q1.size(0), device=q1.device)
            energy = (q1.float() ** 2).sum(dim=1)
            H[diag_indices, diag_indices] = energy

        H[hamming1_mask] = -1.0
        return H

# ====== 4. 主函数 ======
def build_gpu_sparse_matrix(
    ranges_per_group,
    block_size=512,
    matrix_file="sparse_matrix.dat",
    info_file="info.dat",
    connection_ratio=0.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_states = generate_all_states(ranges_per_group).to(device)
    total_states = all_states.size(0)

    # 初始化输出文件
    open(matrix_file, "w").close()

    info_lines = []
    info_lines.append("=== 量子数结构 ===")
    for i, (rn, rm, rk, rs) in enumerate(ranges_per_group):
        info_lines.append(f"组 {i+1}: n∈[0,{rn-1}], m∈[0,{rm-1}], k∈[0,{rk-1}], s∈[0,{rs-1}]")
    info_lines.append(f"总状态数: {total_states}")
    info_lines.append(f"矩阵维度: {total_states} × {total_states}")
    info_lines.append(f"连接保留概率: {connection_ratio}")
    info_lines.append(f"计算设备: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    calculator = HValueCalculatorGPU(connection_ratio=connection_ratio)
    start_time = time.time()
    nonzero_count = 0

    num_blocks_i = (total_states + block_size - 1) // block_size
    total_blocks = num_blocks_i ** 2

    write_buffer = []
    buffer_limit = 8  # 每缓存 8 个 block 数据后写入磁盘

    with torch.no_grad():
        with tqdm(total=total_blocks, desc="计算稀疏矩阵块") as pbar:
            for i_start in range(0, total_states, block_size):
                i_end = min(i_start + block_size, total_states)
                q1 = all_states[i_start:i_end]

                for j_start in range(0, total_states, block_size):
                    j_end = min(j_start + block_size, total_states)
                    q2 = all_states[j_start:j_end]

                    H_block = calculator.compute(q1, q2)
                    nonzero = (H_block != 0)

                    if nonzero.any():
                        nonzero_indices = nonzero.nonzero()
                        i_idx = nonzero_indices[:, 0]
                        j_idx = nonzero_indices[:, 1]

                        i_global = i_start + i_idx
                        j_global = j_start + j_idx
                        values = H_block[i_idx, j_idx]

                        block_data = torch.stack([i_global, j_global, values], dim=1).cpu().numpy()
                        write_buffer.append(block_data)

                        nonzero_count += block_data.shape[0]

                    if len(write_buffer) >= buffer_limit:
                        with open(matrix_file, "ab") as f:
                            np.savetxt(f, np.vstack(write_buffer), fmt="%d %d %.6f")
                        write_buffer.clear()

                    pbar.update(1)

    # 写入剩余缓冲数据
    if write_buffer:
        with open(matrix_file, "ab") as f:
            np.savetxt(f, np.vstack(write_buffer), fmt="%d %d %.6f")
        write_buffer.clear()

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(matrix_file) / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0
    total_possible = total_states * total_states
    sparsity_ratio = nonzero_count / total_possible * 100

    info_lines.append(f"非零元素数量: {nonzero_count}")
    info_lines.append(f"矩阵文件大小: {size_mb:.2f} MB")
    info_lines.append(f"矩阵稀疏率: {sparsity_ratio:.6f}%")
    info_lines.append(f"GPU 显存最高占用: {peak_mem:.2f} MB")
    info_lines.append(f"总耗时: {elapsed:.2f} 秒")

    with open(info_file, "w") as f:
        f.write("\n".join(info_lines))

# ====== 5. 执行入口 ======
if __name__ == "__main__":
    build_gpu_sparse_matrix(
        ranges_per_group=quantum_ranges_per_group,
        block_size=14000,
        matrix_file="sparse_matrix.dat",
        info_file="info.dat",
        connection_ratio=0.005  # 控制为约 0.5% 稀疏率，可调
    )
