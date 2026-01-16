import numpy as np
import matplotlib.pyplot as plt

# ─── Define Primal Points & Line ───
A = np.array([1, 2])
B = np.array([3, 0])
m = (B[1] - A[1]) / (B[0] - A[0])  # slope
b = A[1] - m * A[0]                # intercept

# ─── Define Dual Lines & Point ───
x_dual = np.linspace(-2, 2, 200)
yA = A[0] * x_dual - A[1]    # dual of A: y = a x - b
yB = B[0] * x_dual - B[1]    # dual of B
dual_pt = (m, -b)            # dual of line y = m x + b

# ─── Margins ───
xmin = min(A[0], B[0]) - 0.5
xmax = max(A[0], B[0]) + 0.5
ymin = min(A[1], B[1]) - 0.5
ymax = max(A[1], B[1]) + 0.5

dx_min = x_dual.min() - 0.5
dx_max = x_dual.max() + 0.5
dy_min = min(yA.min(), yB.min()) - 0.5
dy_max = max(yA.max(), yB.max()) + 0.5

# ─── Dual-Vector Points ───
A_dual = np.array([ A[0], -A[1] ])
L_dual = np.array([   m,   -b   ])
B_dual = np.array([ B[0], -B[1] ])
all_dual_x = [A_dual[0], L_dual[0], B_dual[0]]
all_dual_y = [A_dual[1], L_dual[1], B_dual[1]]

# ─── Color Scheme ───
cA, cB, cL = 'C0', 'C1', 'C2'  # A=blue, B=green, L=red

# ─── 1. Primal: Segment ───
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot([A[0], B[0]], [A[1], B[1]], color=cL, lw=2, label='L')
ax.scatter(*A, color=cA, s=30, zorder=3)
ax.text(A[0]+0.1, A[1]+0.1, 'A', color=cA)
ax.scatter(*B, color=cB, s=30, zorder=3)
ax.text(B[0]+0.1, B[1]-0.3, 'B', color=cB)
ax.text((A[0]+B[0])/2 + 0.1, (A[1]+B[1])/2 + 0.1, 'L', color=cL)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_box_aspect(1)
ax.set_title('Primal: Segment')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
fig.savefig('primal_segment.png', dpi=300)
plt.close(fig)

# ─── 2. Dual: Lines ───
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_dual, yA, '--', color=cA, lw=2)
ax.plot(x_dual, yB, '--', color=cB, lw=2)
ax.scatter(*dual_pt, color=cL, s=30, zorder=3)
ax.text(dual_pt[0]+0.05, dual_pt[1]-0.2, r'$(m,-b)$', color=cL)
ax.set_xlim(dx_min, dx_max); ax.set_ylim(dy_min, dy_max)
ax.set_box_aspect(1)
ax.set_title('Dual: Lines')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
fig.savefig('dual_lines.png', dpi=300)
plt.close(fig)

# ─── 3. Primal: Vector ───
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.arrow(
    A[0], A[1],
    B[0] - A[0], B[1] - A[1],
    head_width=0.1,
    length_includes_head=True,
    color=cL,
    lw=2
)
ax.scatter(*A, color=cA, s=30, zorder=3)
ax.text(A[0]+0.1, A[1]+0.1, 'A', color=cA)
ax.scatter(*B, color=cB, s=30, zorder=3)
ax.text(B[0]+0.1, B[1]-0.3, 'B', color=cB)
ax.text((A[0]+B[0])/2 + 0.1, (A[1]+B[1])/2 + 0.1, 'L', color=cL)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_box_aspect(1)
ax.set_title('Primal: Vector')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
fig.savefig('primal_vector.png', dpi=300)
plt.close(fig)

# ─── 4. Dual: Vectors ───
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
# plot dual points
ax.scatter(*A_dual, color=cA, s=30, zorder=3)
ax.text(A_dual[0]+0.1, A_dual[1]+0.1, r'$A^*$', color=cA)
ax.scatter(*L_dual, color=cL, s=30, zorder=3)
ax.text(L_dual[0]-0.4, L_dual[1]+0.2, r'$L^*$', color=cL)
ax.scatter(*B_dual, color=cB, s=30, zorder=3)
ax.text(B_dual[0]+0.1, B_dual[1]+0.1, r'$B^*$', color=cB)
# arrows in dual
ax.arrow(
    A_dual[0], A_dual[1],
    L_dual[0] - A_dual[0], L_dual[1] - A_dual[1],
    head_width=0.1,
    length_includes_head=True,
    color=cA,
    lw=2
)
ax.arrow(
    L_dual[0], L_dual[1],
    B_dual[0] - L_dual[0], B_dual[1] - L_dual[1],
    head_width=0.1,
    length_includes_head=True,
    color=cB,
    lw=2
)
ax.set_xlim(min(all_dual_x)-0.5, max(all_dual_x)+0.5)
ax.set_ylim(min(all_dual_y)-0.5, max(all_dual_y)+0.5)
ax.set_box_aspect(1)
ax.set_title('Dual: Vectors')
ax.set_xlabel('$u$'); ax.set_ylabel('$v$')
fig.savefig('dual_vectors.png', dpi=300)
plt.close(fig)
