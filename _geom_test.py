"""Pure-matplotlib geometry test for the row-2 panel-title layout fix.
No project imports -- just checks the gridspec/text math lines up.
Throwaway file, delete after use.
"""
from matplotlib.figure import Figure

N_COLS = 9
HEIGHT_RATIOS = [2.0, 1.3, 2.0, 2.0]
HSPACE = 0.35

fig = Figure(figsize=(26, 14), dpi=100)
gs = fig.add_gridspec(4, N_COLS, height_ratios=HEIGHT_RATIOS, hspace=HSPACE, wspace=0.08)

ax_bd = fig.add_subplot(gs[0, :])
ax_bd.plot([0, 1, 2], [1, 2, 1])
ax_bd.set_title("B+D% per frame (row 0)")

# max title-line case: worst-case panel gets 4 lines, rest get 2.
line_counts = [2, 2, 2, 2, 3, 4, 3, 2, 2]
max_lines = max(line_counts)

panel_axes = []
for col in range(N_COLS):
    ax = fig.add_subplot(gs[2, col])
    ax.imshow([[0, 1], [1, 0]])
    ax.set_xticks([])
    ax.set_yticks([])
    panel_axes.append(ax)

    n = line_counts[col]
    title_lines = [f"Frame {col - 4:+d}", *[f"stat line {i}" for i in range(n - 1)]]
    start_y = 1.08 + 0.075 * (max_lines - 1)  # fixed across all panels, sized for the worst case
    for line_idx, line in enumerate(title_lines):
        ax.text(0.5, start_y - line_idx * 0.075, line, fontsize=7.5, ha="center", va="bottom",
                transform=ax.transAxes, clip_on=False)

ax_vm = fig.add_subplot(gs[3, :])
ax_vm.plot([0, 1, 2], [1, 2, 1])
ax_vm.set_title("Vm traces (row 3)")

bottoms, tops, _, _ = gs.get_grid_positions(fig)
print("tops", tops)
print("bottoms", bottoms)
h2 = tops[2] - bottoms[2]
settings_y = tops[2] + (start_y - 1.0) * h2 + 0.02
print("h2", h2, "settings_y (fig frac)", settings_y, "bottoms[0]", bottoms[0])
fig.text(0.5, settings_y, "RegionAnalyzer -- OBJ=10X min_samples=15 found 43 -> kept 1", ha="center", va="bottom",
          fontsize=12, fontweight="bold")

out = r"D:\MyDB\2_Programs\PG_005\_geom_test.png"
fig.savefig(out)
print("saved", out)
