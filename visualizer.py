import matplotlib.pyplot as plt
import numpy as np
from vector_engine import dot_product, cross_product_2d
import math

def visualize_vectors(X, Y, title_suffix=""):
    """
    Visualizes vectors X and Y, their dot product, cross product,
    and the parallelogram they form, along with geometric interpretations.
    Optionally adds a suffix to the plot title.
    """
    if len(X) != 2 or len(Y) != 2:
        raise ValueError("Input vectors must be 2-dimensional.")

    dp = dot_product(X, Y)
    cp = cross_product_2d(X, Y)
    mag_X = math.sqrt(X[0]**2 + X[1]**2)
    mag_Y = math.sqrt(Y[0]**2 + Y[1]**2)
    angle_deg = 0
    if mag_X * mag_Y != 0:
        cos_theta = max(-1, min(1, dp / (mag_X * mag_Y)))
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

    fig, ax = plt.subplots()
    ax.arrow(0, 0, X[0], X[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label=f'X = {X}', length_includes_head=True)
    ax.arrow(0, 0, Y[0], Y[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label=f'Y = {Y}', length_includes_head=True)

    para_verts = np.array([[0, 0], X, [X[0] + Y[0], X[1] + Y[1]], Y, [0,0]])
    parallelogram = plt.Polygon(para_verts[:4,:], closed=True, fill=True, edgecolor='gray', facecolor='lightgray', alpha=0.5)
    ax.add_patch(parallelogram)

    center_x = (X[0] + Y[0]) / 2
    center_y = (X[1] + Y[1]) / 2
    ax.text(center_x, center_y, f'Area = {cp}', ha='center', va='center', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=0.2))

    all_points = np.array([[0,0], X, Y, [X[0]+Y[0], X[1]+Y[1]]])
    plot_margin = 1.5
    # Adjust margin if vectors are very small or zero to avoid tiny plots
    if mag_X < 0.1 and mag_Y < 0.1 :
        plot_margin = 0.5

    min_x = min(all_points[:,0]) - plot_margin
    max_x = max(all_points[:,0]) + plot_margin
    min_y = min(all_points[:,1]) - plot_margin
    max_y = max(all_points[:,1]) + plot_margin

    # Ensure min and max are different, especially for zero vectors
    if min_x == max_x:
        max_x += 0.5
        min_x -= 0.5
    if min_y == max_y:
        max_y += 0.5
        min_y -= 0.5

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(f"2D Vector Operations{title_suffix}")
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    dot_product_interpretation = (
        f"X · Y = {dp}\n"
        f"  - Measures how much one vector extends in the direction of the other.\n"
        f"  - Angle between X and Y: θ ≈ {angle_deg:.2f}°\n"
        f"  - If X·Y > 0, angle < 90° (acute)\n"
        f"  - If X·Y < 0, angle > 90° (obtuse)\n"
        f"  - If X·Y = 0, angle = 90° (orthogonal)"
    )
    cross_product_interpretation = (
        f"X × Y (z-comp) = {cp}\n"
        f"  - Represents the signed area of the parallelogram formed by X and Y.\n"
        f"  - Area = |{cp}|\n"
        f"  - Sign indicates orientation:\n"
        f"    - Positive: Y is counter-clockwise from X.\n"
        f"    - Negative: Y is clockwise from X.\n"
        f"    - Zero: X and Y are collinear."
    )
    info_text = f"{dot_product_interpretation}\n\n{cross_product_interpretation}"
    fig.text(0.01, 0.01, info_text, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    ax.legend(loc='upper right')
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)
    plt.show()

if __name__ == "__main__":
    test_cases = [
        {"X": [2, 3], "Y": [4, 1], "title": " (Case 1: General Vectors)"},
        {"X": [4, 1], "Y": [2, 3], "title": " (Case 2: Swapped Vectors - Negative Area)"}, # Should give negative cross product
        {"X": [2, 2], "Y": [-2, 2], "title": " (Case 3: Orthogonal Vectors)"}, # Dot product should be 0
        {"X": [3, 1], "Y": [6, 2], "title": " (Case 4: Collinear Vectors)"},    # Cross product should be 0
        {"X": [-1, -2], "Y": [-3, 1], "title": " (Case 5: Negative Components)"},
        {"X": [0, 0], "Y": [2, 3], "title": " (Case 6: Zero Vector X)"},
        {"X": [2, 3], "Y": [0, 0], "title": " (Case 7: Zero Vector Y)"}
    ]

    for i, case in enumerate(test_cases):
        print(f"--- Running Test Case {i+1} ---")
        print(f"Input Vector X: {case['X']}")
        print(f"Input Vector Y: {case['Y']}")

        try:
            visualize_vectors(case["X"], case["Y"], title_suffix=case["title"])
        except ValueError as e:
            print(f"Error: {e}")
        except ImportError:
            print("Error: matplotlib/numpy is required. Please install them (e.g., pip install matplotlib numpy)")
        print("-" * 30 + "\n")
