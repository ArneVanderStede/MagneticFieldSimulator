import numpy as np
import time

BENCHMARK = False

# benchmark settings
BENCH_GRIDS   = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
BENCH_REPEATS = 3
BENCH_WIRES   = ["straight", "loop", "solenoid"]


MU0              = 4e-7 * np.pi
EXTENT           = 1.0
LOOP_RADIUS      = 0.5
SOLENOID_RADIUS  = 0.35
SOLENOID_LENGTH  = 1.2
SOLENOID_TURNS   = 4

# GPU detection
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    _ = cp.array([1.0], dtype=cp.float32)
    HAS_GPU   = True
    gpu_props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME  = gpu_props["name"].decode() if isinstance(gpu_props["name"], bytes) else str(gpu_props["name"])
except Exception:
    HAS_GPU  = False
    cp       = None
    GPU_NAME = "Not available"

# CUDA kernel
_CUDA_SRC = r"""
extern "C" __global__
void biot_savart_2d(
    const float* __restrict__ pts,
    const float* __restrict__ Ax,
    const float* __restrict__ Ay,
    const float* __restrict__ Az,
    const float* __restrict__ Bx,
    const float* __restrict__ By,
    const float* __restrict__ Bz,
    float*       __restrict__ Bout,
    const int num_points, const int num_segments,
    const float prefactor, const float mask_radius)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    const float pt_x = pts[point_idx*2];
    const float pt_y = pts[point_idx*2+1];
    const float pt_z = 0.f;
    float field_x = 0.f, field_y = 0.f;
    float min_dist_sq = 1e30f;

    for (int k = 0; k < num_segments; ++k) {
        const float start_x = Ax[k], start_y = Ay[k], start_z = Az[k];
        const float end_x   = Bx[k], end_y   = By[k], end_z   = Bz[k];

        const float to_start_x = pt_x-start_x, to_start_y = pt_y-start_y, to_start_z = pt_z-start_z;
        const float to_end_x   = pt_x-end_x,   to_end_y   = pt_y-end_y,   to_end_z   = pt_z-end_z;

        const float cross_x = to_start_y*to_end_z - to_start_z*to_end_y;
        const float cross_y = to_start_z*to_end_x - to_start_x*to_end_z;

        const float dist_start = sqrtf(to_start_x*to_start_x + to_start_y*to_start_y + to_start_z*to_start_z);
        const float dist_end   = sqrtf(to_end_x*to_end_x     + to_end_y*to_end_y     + to_end_z*to_end_z);
        const float dot_prod   = to_start_x*to_end_x + to_start_y*to_end_y + to_start_z*to_end_z;
        const float denom      = dist_start * dist_end * (dist_start*dist_end + dot_prod) + 1e-30f;
        const float factor     = (dist_start + dist_end) / denom;

        field_x += cross_x * factor;
        field_y += cross_y * factor;

        const float seg_dx     = end_x-start_x, seg_dy = end_y-start_y, seg_dz = end_z-start_z;
        const float seg_len_sq = seg_dx*seg_dx + seg_dy*seg_dy + seg_dz*seg_dz + 1e-30f;
        float proj_t = ((pt_x-start_x)*seg_dx + (pt_y-start_y)*seg_dy + (pt_z-start_z)*seg_dz) / seg_len_sq;
        proj_t = fminf(1.f, fmaxf(0.f, proj_t));
        const float diff_x  = pt_x - (start_x + proj_t*seg_dx);
        const float diff_y  = pt_y - (start_y + proj_t*seg_dy);
        const float diff_z  = pt_z - (start_z + proj_t*seg_dz);
        const float dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
        if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
    }

    const float wire_mask = (sqrtf(min_dist_sq) < mask_radius) ? 0.f : 1.f;
    Bout[point_idx*2]   = prefactor * field_x * wire_mask;
    Bout[point_idx*2+1] = prefactor * field_y * wire_mask;
}
"""

_kernel = None
def _get_kernel():
    global _kernel
    if _kernel is None and HAS_GPU:
        _kernel = cp.RawKernel(_CUDA_SRC, "biot_savart_2d")
    return _kernel

def _warmup_gpu():
    if not HAS_GPU:
        return
    zero_components  = np.zeros(2, dtype=np.float32)
    seg_start_z      = np.array([-0.1, 0.0], dtype=np.float32)
    seg_end_z        = np.array([ 0.0, 0.1], dtype=np.float32)
    dummy_points_gpu = cp.zeros((4, 2), dtype=cp.float32)
    dummy_output_gpu = cp.zeros((4, 2), dtype=cp.float32)
    _get_kernel()((1,), (4,),
        (dummy_points_gpu.ravel(),
         cp.asarray(zero_components), cp.asarray(zero_components), cp.asarray(seg_start_z),
         cp.asarray(zero_components), cp.asarray(zero_components), cp.asarray(seg_end_z),
         dummy_output_gpu.ravel(),
         np.int32(4), np.int32(2), np.float32(1e-7), np.float32(0.04)))
    cp.cuda.Device().synchronize()

def make_segments(wire, num_segments=300):
    if wire == "loop":
        theta     = np.linspace(0, 2*np.pi, num_segments+1, dtype=np.float32)
        x         = LOOP_RADIUS * np.cos(theta)
        y         = LOOP_RADIUS * np.sin(theta)
        z         = np.zeros(num_segments+1, dtype=np.float32)
        seg_start = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
        seg_end   = np.stack([x[1: ], y[1: ], z[1: ]], axis=1)

    elif wire == "solenoid":
        total_pts = num_segments + 1
        # Parametric helix: t ∈ [0, 2π * N_turns]
        t         = np.linspace(0, 2 * np.pi * SOLENOID_TURNS, total_pts, dtype=np.float32)
        x         = SOLENOID_RADIUS * np.cos(t)
        y         = SOLENOID_RADIUS * np.sin(t)
        z         = np.linspace(-SOLENOID_LENGTH / 2, SOLENOID_LENGTH / 2, total_pts, dtype=np.float32)
        seg_start = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
        seg_end   = np.stack([x[1: ], y[1: ], z[1: ]], axis=1)

    else:  # straight
        z         = np.linspace(-1.5, 1.5, num_segments+1, dtype=np.float32)
        seg_start = np.stack([np.zeros(num_segments), np.zeros(num_segments), z[:-1]], axis=1)
        seg_end   = np.stack([np.zeros(num_segments), np.zeros(num_segments), z[1: ]], axis=1)

    return seg_start.astype(np.float32), seg_end.astype(np.float32)

# CPU Biot-Savart
def bs_cpu(I, gx, gy, wire="straight", num_segments=300, mask_r=0.04):
    seg_start, seg_end = make_segments(wire, num_segments)
    seg_vec = seg_end - seg_start
    points  = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    vec_to_start  = points[:, None, :] - seg_start[None]
    vec_to_end    = points[:, None, :] - seg_end[None]
    cross_product = np.cross(vec_to_start, vec_to_end, axis=2)
    dist_to_start = np.linalg.norm(vec_to_start, axis=2)
    dist_to_end   = np.linalg.norm(vec_to_end,   axis=2)
    dot_product   = np.einsum("ijk,ijk->ij", vec_to_start, vec_to_end)
    denominator   = dist_to_start * dist_to_end * (dist_to_start * dist_to_end + dot_product) + 1e-30
    scalar_factor = (dist_to_start + dist_to_end) / denominator
    prefactor     = MU0 * I / (4 * np.pi)
    B_field       = prefactor * np.sum(cross_product * scalar_factor[:, :, None], axis=1)

    seg_len_sq     = np.einsum("ij,ij->i", seg_vec, seg_vec) + 1e-30
    proj_param     = np.einsum("mij,ij->mi", points[:, None, :] - seg_start[None], seg_vec) / seg_len_sq
    proj_param     = np.clip(proj_param, 0.0, 1.0)
    nearest_points = seg_start[None] + proj_param[:, :, None] * seg_vec[None]
    dist_sq        = np.sum((points[:, None, :] - nearest_points)**2, axis=2)
    B_field[np.sqrt(np.min(dist_sq, axis=1)) < mask_r] = 0.0

    grid_shape = gx.shape
    return B_field[:, 0].reshape(grid_shape), B_field[:, 1].reshape(grid_shape)

# GPU Biot-Savart
def bs_gpu(I, gx, gy, wire="straight", num_segments=300, mask_r=0.04):
    seg_start, seg_end = make_segments(wire, num_segments)
    num_points = gx.size
    grid_xy    = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    points_gpu      = cp.ascontiguousarray(cp.asarray(grid_xy))
    field_gpu       = cp.zeros((num_points, 2), dtype=cp.float32)
    seg_start_x_gpu = cp.asarray(seg_start[:, 0])
    seg_start_y_gpu = cp.asarray(seg_start[:, 1])
    seg_start_z_gpu = cp.asarray(seg_start[:, 2])
    seg_end_x_gpu   = cp.asarray(seg_end[:, 0])
    seg_end_y_gpu   = cp.asarray(seg_end[:, 1])
    seg_end_z_gpu   = cp.asarray(seg_end[:, 2])

    prefactor         = np.float32(MU0 * I / (4 * np.pi))
    threads_per_block = 256
    num_blocks        = (num_points + threads_per_block - 1) // threads_per_block

    _get_kernel()(
        (num_blocks,), (threads_per_block,),
        (points_gpu.ravel(),
         seg_start_x_gpu, seg_start_y_gpu, seg_start_z_gpu,
         seg_end_x_gpu,   seg_end_y_gpu,   seg_end_z_gpu,
         field_gpu.ravel(), np.int32(num_points), np.int32(num_segments), prefactor, np.float32(mask_r))
    )
    cp.cuda.Device().synchronize()

    field_result = cp.asnumpy(field_gpu)
    grid_shape   = gx.shape
    return field_result[:, 0].reshape(grid_shape), field_result[:, 1].reshape(grid_shape)

# 
#  BENCHMARK MODE
# 
def run_benchmark():
    if HAS_GPU:
        print("  [warming up GPU kernel…]")
        _warmup_gpu()

    for wire in BENCH_WIRES:
        print()
        print("=" * 65)
        print(f"  Wire type : {wire.upper()}   S=300 segments")
        print(f"  CPU       : NumPy")
        print(f"  GPU       : {GPU_NAME}")
        print(f"  Runs      : best of {BENCH_REPEATS}")
        print("=" * 65)
        print(f"{'Grid':>6}  {'Points':>8}  {'Ops':>12}  "
              f"{'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>8}")
        print("-" * 65)

        for grid_size in BENCH_GRIDS:
            coords     = np.linspace(-EXTENT, EXTENT, grid_size)
            gx, gy     = np.meshgrid(coords, coords)
            num_points = grid_size * grid_size
            num_ops    = num_points * 300

            cpu_times = []
            for _ in range(BENCH_REPEATS):
                t0 = time.perf_counter()
                bs_cpu(1.0, gx, gy, wire=wire)
                cpu_times.append((time.perf_counter() - t0) * 1e3)
            cpu_ms = min(cpu_times)

            if HAS_GPU:
                gpu_times = []
                for _ in range(BENCH_REPEATS):
                    t0 = time.perf_counter()
                    bs_gpu(1.0, gx, gy, wire=wire)
                    gpu_times.append((time.perf_counter() - t0) * 1e3)
                gpu_ms  = min(gpu_times)
                speedup = cpu_ms / gpu_ms
                print(f"{grid_size:>4}²  {num_points:>8,}  {num_ops:>12,}  "
                      f"{cpu_ms:>10.1f}  {gpu_ms:>10.1f}  {speedup:>7.1f}x")
            else:
                print(f"{grid_size:>4}²  {num_points:>8,}  {num_ops:>12,}  "
                      f"{cpu_ms:>10.1f}  {'N/A':>10}  {'N/A':>8}")

        print("=" * 65)

# 
#  GRAPHICAL MODE
# 
def run_gui():
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib import cm, colors as mcolors

    def draw_quiver(ax, gx, gy, Bx, By, skip=1, arrow_len=0.10):
        step_slice     = slice(None, None, skip)
        gx_sub, gy_sub = gx[step_slice, step_slice], gy[step_slice, step_slice]
        Bx_sub, By_sub = Bx[step_slice, step_slice], By[step_slice, step_slice]
        B_magnitude    = np.sqrt(Bx_sub**2 + By_sub**2)
        nonzero_B      = B_magnitude[B_magnitude > 0]
        vmin = float(nonzero_B.min())              if len(nonzero_B) else 1e-9
        vmax = float(np.percentile(nonzero_B, 98)) if len(nonzero_B) else 1e-6
        B_mag_safe  = B_magnitude + 1e-30
        log_B_flat  = np.log1p(B_mag_safe.ravel())
        arrow_scale = 0.15 + 0.85*(log_B_flat - log_B_flat.min())/(log_B_flat.max() - log_B_flat.min() + 1e-20)
        arrow_scale = arrow_scale.reshape(B_mag_safe.shape)
        arrow_dx    = Bx_sub / B_mag_safe * arrow_scale * arrow_len
        arrow_dy    = By_sub / B_mag_safe * arrow_scale * arrow_len
        log_norm    = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        arrow_colors = cm.plasma(np.clip(log_norm(B_magnitude.ravel()), 0, 1))
        ax.quiver(gx_sub, gy_sub, arrow_dx, arrow_dy, color=arrow_colors,
                  scale=1.0, scale_units="xy", angles="xy",
                  width=0.003, headwidth=4, headlength=5, alpha=0.92)
        return log_norm, B_magnitude

    root = tk.Tk()
    root.title("2D Biot-Savart Simulator")
    bg_color = "#f4f4f4"
    root.configure(bg=bg_color)

    frame_plot = tk.Frame(root, bg="white")
    frame_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig = Figure(figsize=(6, 6), facecolor="white")
    ax  = fig.add_subplot(111, aspect="equal")
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.08, top=0.92)

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, frame_plot).update()

    colorbar_mappable = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="plasma")
    colorbar_mappable.set_array([])
    cbar = fig.colorbar(colorbar_mappable, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("|B| (T)", fontsize=8)

    frame_ctrl = tk.Frame(root, bg=bg_color, width=240)
    frame_ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
    frame_ctrl.pack_propagate(False)

    def section(title):
        label_frame = ttk.LabelFrame(frame_ctrl, text=title, padding="6 4")
        label_frame.pack(fill=tk.X, pady=4)
        return label_frame

    def make_slider(parent, label, var, lo, hi, res):
        row = tk.Frame(parent, bg=bg_color); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, width=12, anchor="w",
                 bg=bg_color, font=("Helvetica", 9)).pack(side=tk.LEFT)
        tk.Label(row, textvariable=var, width=6, anchor="e",
                 bg=bg_color, font=("Helvetica", 9)).pack(side=tk.RIGHT)
        tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                 orient=tk.HORIZONTAL, bg=bg_color, highlightthickness=0,
                 showvalue=False, length=80).pack(side=tk.LEFT, fill=tk.X, expand=True)

    backend_section  = section("Backend")
    gpu_status_color = "#006600" if HAS_GPU else "#990000"
    tk.Label(backend_section, text=f"GPU: {GPU_NAME}",
             fg=gpu_status_color, bg=bg_color, font=("Helvetica", 8)).pack(anchor="w")
    BACKENDS    = ["CPU – NumPy"] + (["GPU – CUDA"] if HAS_GPU else [])
    backend_var = tk.StringVar(value=BACKENDS[-1] if HAS_GPU else BACKENDS[0])
    ttk.Combobox(backend_section, textvariable=backend_var, values=BACKENDS,
                 state="readonly", width=22, font=("Helvetica", 9)
                 ).pack(fill=tk.X, pady=(4, 0))

    wire_section = section("Wire Type")
    wire_var = tk.StringVar(value="straight")
    for w in ["straight", "loop", "solenoid"]:
        tk.Radiobutton(wire_section, text=w.capitalize(), variable=wire_var, value=w,
                       bg=bg_color, font=("Helvetica", 9),
                       command=lambda: _debounce()).pack(anchor="w")

    current_section = section("Current")
    I_mag_var = tk.DoubleVar(value=1.0)
    make_slider(current_section, "I  (A)", I_mag_var, 0.5, 20.0, 0.5)
    I_sign = [+1]
    direction_label_var = tk.StringVar(value="Direction:  ↑  (out of page)")

    def toggle_dir():
        I_sign[0] *= -1
        direction_label_var.set("Direction:  ↑  (out of page)" if I_sign[0] > 0
                                else "Direction:  ↓  (into page)")
        direction_button.config(bg="#e8f0e8" if I_sign[0] > 0 else "#f0e8e8")
        _debounce()

    direction_button = tk.Button(current_section, textvariable=direction_label_var, bg="#e8f0e8",
                                 relief="groove", font=("Helvetica", 9),
                                 cursor="hand2", command=toggle_dir)
    direction_button.pack(fill=tk.X, pady=(6, 0))

    computation_section = section("Computation")
    pts_var  = tk.IntVar(value=30)
    skip_var = tk.IntVar(value=1)
    make_slider(computation_section, "Grid pts",   pts_var,  10, 200, 5)
    make_slider(computation_section, "Arrow skip", skip_var,  1,   8, 1)

    timing_section = section("Timing")
    cpu_time_label = tk.Label(timing_section, text="CPU  —", bg=bg_color, font=("Courier", 10),
                              fg="#004488", anchor="w")
    cpu_time_label.pack(fill=tk.X)
    gpu_time_label = tk.Label(timing_section, text="GPU  —", bg=bg_color, font=("Courier", 10),
                              fg="#006600", anchor="w")
    gpu_time_label.pack(fill=tk.X)
    speedup_label = tk.Label(timing_section, text="", bg=bg_color, font=("Courier", 9),
                             fg="#880000", anchor="w")
    speedup_label.pack(fill=tk.X)
    points_info_label = tk.Label(timing_section, text="", bg=bg_color, font=("Courier", 8),
                                 fg="#444", anchor="w")
    points_info_label.pack(fill=tk.X)

    last_timing_ms   = {"CPU": None, "GPU": None}
    pending_after_id = [None]

    def _debounce(ms=250):
        if pending_after_id[0]:
            try: root.after_cancel(pending_after_id[0])
            except: pass
        pending_after_id[0] = root.after(ms, refresh)

    def refresh():
        wire_type = wire_var.get()
        I         = float(I_mag_var.get()) * I_sign[0]
        grid_size = int(pts_var.get())
        skip      = int(skip_var.get())
        use_gpu   = backend_var.get().startswith("GPU") and HAS_GPU

        coords = np.linspace(-EXTENT, EXTENT, grid_size)
        gx, gy = np.meshgrid(coords, coords)

        t0 = time.perf_counter()
        if use_gpu:
            Bx, By     = bs_gpu(I, gx, gy, wire=wire_type)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            last_timing_ms["GPU"] = elapsed_ms
            gpu_time_label.config(text=f"GPU  {elapsed_ms:8.1f} ms")
        else:
            Bx, By     = bs_cpu(I, gx, gy, wire=wire_type)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            last_timing_ms["CPU"] = elapsed_ms
            cpu_time_label.config(text=f"CPU  {elapsed_ms:8.1f} ms")

        if last_timing_ms["CPU"] and last_timing_ms["GPU"]:
            speedup_label.config(text=f"Speedup  x{last_timing_ms['CPU']/last_timing_ms['GPU']:.1f}")
        else:
            speedup_label.config(text="Switch backends to compare")

        ax.cla()
        draw_quiver(ax, gx, gy, Bx, By, skip=skip)

        if wire_type == "loop":
            theta = np.linspace(0, 2*np.pi, 300)
            ax.plot(LOOP_RADIUS*np.cos(theta), LOOP_RADIUS*np.sin(theta),
                    "-", color="red", lw=2.0, alpha=0.85, zorder=5, label="Loop wire")
            ax.legend(fontsize=7, loc="upper right")
        elif wire_type == "solenoid":
            theta = np.linspace(0, 2*np.pi, 300)
            ax.plot(SOLENOID_RADIUS*np.cos(theta), SOLENOID_RADIUS*np.sin(theta),
                    "--", color="red", lw=1.8, alpha=0.85, zorder=5, label="Solenoid (xy cross-section)")
            # mark the turn entry/exit points where the helix pierces z=0
            n_turns = SOLENOID_TURNS
            for k in range(n_turns + 1):
                phi = 2 * np.pi * k
                ax.plot(SOLENOID_RADIUS * np.cos(phi), SOLENOID_RADIUS * np.sin(phi),
                        "o", color="red", ms=5, mec="black", mew=0.6, zorder=6)
            ax.legend(fontsize=7, loc="upper right")
        else:
            wire_symbol = "⊙" if I_sign[0] > 0 else "⊗"
            ax.plot(0, 0, "o", color="red", ms=14, mec="black", mew=0.8, zorder=5)
            ax.text(0, 0, wire_symbol, ha="center", va="center",
                    fontsize=14, color="white", zorder=6, fontweight="bold")

        B_magnitude = np.sqrt(Bx**2 + By**2)
        nonzero_B   = B_magnitude[B_magnitude > 0]
        if len(nonzero_B):
            scalar_mappable = cm.ScalarMappable(
                norm=mcolors.LogNorm(vmin=float(nonzero_B.min()),
                                     vmax=float(np.percentile(nonzero_B, 98))),
                cmap="plasma")
            scalar_mappable.set_array([])
            cbar.update_normal(scalar_mappable)
        cbar.set_label("|B| (T)", fontsize=8)

        ax.set_xlim(-EXTENT, EXTENT); ax.set_ylim(-EXTENT, EXTENT)
        ax.set_xlabel("x [m]", fontsize=9); ax.set_ylabel("y [m]", fontsize=9)
        wire_label      = (f"Loop R={LOOP_RADIUS}m" if wire_type == "loop" else
                           f"Solenoid R={SOLENOID_RADIUS}m N={SOLENOID_TURNS}" if wire_type == "solenoid" else
                           "Straight wire")
        backend_label   = "GPU" if use_gpu else "CPU"
        direction_label = "↑ out" if I_sign[0] > 0 else "↓ in"
        ax.set_title(f"{wire_label}   I={I:+.1f}A ({direction_label})   {grid_size}×{grid_size}   [{backend_label}]",
                     fontsize=9)
        ax.set_facecolor("#fafafa")
        ax.grid(True, color="#ddd", linewidth=0.4)
        ax.set_aspect("equal")

        num_points = grid_size * grid_size
        points_info_label.config(text=f"{num_points:,} pts × 300 segs = {num_points*300:,} ops")
        canvas.draw_idle()

    for v in (backend_var, wire_var, I_mag_var, pts_var, skip_var):
        v.trace_add("write", lambda *_: _debounce())

    if HAS_GPU:
        _warmup_gpu()
    refresh()
    root.mainloop()

if __name__ == "__main__":
    if BENCHMARK:
        run_benchmark()
    else:
        run_gui()