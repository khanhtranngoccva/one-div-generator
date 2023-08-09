import math

import matplotlib.pyplot

Point = tuple[int | float, int | float]


def circumcenter(v1, v2, v3):
    ax, ay = v1
    bx, by = v2
    cx, cy = v3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


def centroid(v1, v2, v3):
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    return (x1 + x2 + x3) // 3, (y1 + y2 + y3) // 3


def get_bounding_box(v1: Point, v2: Point, v3: Point):
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    min_x = min([x1, x2, x3]) - 0
    max_x = max([x1, x2, x3]) + 1
    min_y = min([y1, y2, y3]) - 0
    max_y = max([y1, y2, y3]) + 1

    return min_x, min_y, max_x - min_x, max_y - min_y


def handle_triangle_above(v1, v2, v3, color_hex: str, edge_angle=0.005, rounding=2):
    v1, v2 = sorted((v1, v2), key=lambda p: p[0])

    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    x, y, w, h = get_bounding_box((x1, y1), (x2, y2), (x3, y3))

    x_ratio = (x3 - x) / w * 100
    starting_angle = math.pi - math.atan((x2 - x3) / h) - edge_angle
    ending_angle = math.pi - math.atan((x1 - x3) / h)
    angle_diff = ending_angle - starting_angle + 2 * edge_angle

    out = f'conic-gradient(from {round(starting_angle, rounding)}rad at {round(x_ratio, rounding)}% 0%, {color_hex} {round(angle_diff, rounding)}rad, transparent {round(angle_diff, rounding)}rad) {round(x, rounding)}px {round(y, rounding)}px / {round(w, rounding)}px {round(h, rounding)}px'
    return out


def handle_triangle_below(v1, v2, v3, color_hex: str, edge_angle=0.005, rounding=2):
    v1, v2 = sorted((v1, v2), key=lambda p: p[0])

    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    x, y, w, h = get_bounding_box((x1, y1), (x2, y2), (x3, y3))

    x_ratio = (x3 - x) / w * 100
    starting_angle = math.atan((x1 - x3) / h) - edge_angle
    ending_angle = math.atan((x2 - x3) / h)
    angle_diff = ending_angle - starting_angle + 2 * edge_angle

    out = f'conic-gradient(from {round(starting_angle, rounding)}rad at {round(x_ratio, rounding)}% 100%, {color_hex} {round(angle_diff, rounding)}rad, transparent {round(angle_diff, rounding)}rad) {round(x, rounding)}px {round(y, rounding)}px / {round(w, rounding)}px {round(h, rounding)}px'
    return out


def convert_8bit_to_hex(data: int):
    hex_string = hex(math.floor(data))[2:].zfill(2)
    return hex_string


def color_to_hex(data: tuple[int, int, int]):
    hex_string = "".join(map(convert_8bit_to_hex, data))
    return f'#{hex_string}'


def convert_triangle(triangle: list[tuple[int, int]], color: tuple[int, int, int], edge_angle=0.03, rounding=2):
    v1, v2, v3 = sorted(triangle, key=lambda x: x[1])
    color_hex = color_to_hex(color)

    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    if y1 == y2 == y3:
        gradients = []
    elif y1 == y2:
        gradients = [handle_triangle_below(v1, v2, v3, color_hex, edge_angle, rounding)]
    elif y2 == y3:
        gradients = [handle_triangle_above(v2, v3, v1, color_hex, edge_angle, rounding)]
    else:
        ratio1 = (y2 - y1) / (y3 - y1)
        y4 = y2
        if x3 > x1:
            x4 = x1 + ratio1 * (x3 - x1)
        else:
            x4 = x1 + ratio1 * (x3 - x1)
        v4 = (x4, y4)
        # matplotlib.pyplot.plot([x1, x2, x3, x4], [y1, y2, y3, y4])
        # matplotlib.pyplot.show()

        gradients = [
            handle_triangle_above(v2, v4, v1, color_hex, edge_angle, rounding),
            handle_triangle_below(v2, v4, v3, color_hex, edge_angle, rounding),
        ]
    return gradients
