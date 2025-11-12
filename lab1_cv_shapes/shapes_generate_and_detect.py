import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def create_soft_polygon(ax, origin, size, sides, color, smoothing=0.1, thickness=2):
    shift = np.pi / sides
    angle_array = np.linspace(0, 2 * np.pi, sides, endpoint=False) + shift
    points = np.stack((np.cos(angle_array), np.sin(angle_array)), axis=1) * size + origin

    path_pts = []
    path_cmds = []
    smooth_radius = size * smoothing

    for idx in range(sides):
        prev = points[(idx - 1) % sides]
        current = points[idx]
        next_ = points[(idx + 1) % sides]

        in_vec = (current - prev)
        in_vec = in_vec / np.linalg.norm(in_vec) * smooth_radius

        out_vec = (next_ - current)
        out_vec = out_vec / np.linalg.norm(out_vec) * smooth_radius

        corner_start = current - in_vec
        corner_end = current + out_vec

        if idx == 0:
            path_pts.append(corner_start)
            path_cmds.append(Path.MOVETO)
        else:
            path_pts.append(corner_start)
            path_cmds.append(Path.LINETO)

        path_pts.append(current)
        path_cmds.append(Path.CURVE3)
        path_pts.append(corner_end)
        path_cmds.append(Path.CURVE3)

    path_pts.append(path_pts[0])
    path_cmds.append(Path.LINETO)

    shape_path = Path(path_pts, path_cmds)
    patch = PathPatch(shape_path, facecolor=color, edgecolor=color, lw=thickness)
    ax.add_patch(patch)

def create_star_shape(ax, center, size, color, softness=0.12, stroke=2):
    star_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = np.tile([size, size * 0.5], 5)
    star_points = np.stack((np.cos(star_angles) * radii, np.sin(star_angles) * radii), axis=1) + center

    bezier_points = []
    bezier_cmds = []
    r = size * softness

    for i in range(10):
        a = star_points[(i - 1) % 10]
        b = star_points[i]
        c = star_points[(i + 1) % 10]

        to_a = (b - a) / np.linalg.norm(b - a) * r
        to_c = (c - b) / np.linalg.norm(c - b) * r

        start = b - to_a
        end = b + to_c

        if i == 0:
            bezier_points.append(start)
            bezier_cmds.append(Path.MOVETO)
        else:
            bezier_points.append(start)
            bezier_cmds.append(Path.LINETO)

        bezier_points.append(b)
        bezier_cmds.append(Path.CURVE3)
        bezier_points.append(end)
        bezier_cmds.append(Path.CURVE3)

    bezier_points.append(bezier_points[0])
    bezier_cmds.append(Path.LINETO)

    final_path = Path(bezier_points, bezier_cmds)
    patch = PathPatch(final_path, facecolor=color, edgecolor=color, lw=stroke)
    ax.add_patch(patch)

def does_overlap(new_c, new_r, existing, gap=1.15):
    for (c, r) in existing:
        if np.linalg.norm(np.array(c) - np.array(new_c)) < (r + new_r) * gap:
            return True
    return False

def visualize_and_detect():
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_facecolor("black")

    shape_types = ["star", "triangle", "square", "pentagon"]
    color_options = {"blue": "blue", "yellow": "yellow", "red": "red"}
    used_positions = []

    for shape in shape_types:
        for name, color in color_options.items():
            while True:
                pos = np.random.uniform(-1.5, 1.5, size=2)
                rad = np.random.uniform(0.2, 0.35)
                if not does_overlap(pos, rad, used_positions):
                    used_positions.append((pos, rad))
                    break

            if shape == "star":
                create_star_shape(ax, pos, rad, color)
            else:
                sides = {"triangle": 3, "square": 4, "pentagon": 5}[shape]
                create_soft_polygon(ax, pos, rad, sides, color)

    img_path = "generated_output.png"
    plt.savefig(img_path, facecolor='black', bbox_inches='tight')
    plt.close()

    img = cv2.imread(img_path)
    cv2.imshow("Artwork", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Изображение загружено, переход к анализу...")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    selected = 1  # 1 - жёлтый, 2 - красный, 3 - синий

    bounds = {
        1: (np.array([20, 100, 100]), np.array([30, 255, 255])),
        2: (np.array([0, 100, 100]), np.array([10, 255, 255])),
        3: (np.array([100, 100, 100]), np.array([130, 255, 255]))
    }

    lower, upper = bounds.get(selected, (None, None))
    if lower is None:
        raise ValueError("Недопустимый выбор цвета")

    mask = cv2.inRange(hsv, lower, upper)
    isolated = cv2.bitwise_and(img, img, mask=mask)

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blur, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 3:
            cv2.drawContours(result, [approx], -1, (255, 0, 255), 3)

    cv2.imwrite("Source/mask.png", mask)
    cv2.imwrite("Source/selected.png", isolated)
    cv2.imwrite("Source/contours.png", result)

    cv2.imshow("Маска", mask)
    cv2.imshow("Выделение", isolated)
    cv2.imshow("Контуры", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Завершено.")

visualize_and_detect()
