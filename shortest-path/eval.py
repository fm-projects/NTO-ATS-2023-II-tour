import cv2
from queue import PriorityQueue


G = {
    1: {3: 13, 11: 14},
    3: {5: 13, 16: 14},
    5: {7: 13, 10: 14},
    7: {1: 13, 13: 14},
    2: {8: 13, 13: 9},
    4: {2: 13, 11: 9},
    6: {4: 13, 16: 9},
    8: {6: 13, 10: 9},
    9: {2: 9, 3: 14},
    12: {6: 9, 7: 14},
    14: {4: 9, 5: 14},
    15: {8: 9, 1: 14},
    10: {9: 13, 15: 9, 14: 14},
    11: {12: 13, 14: 9, 15: 14},
    13: {14: 13, 9: 9, 12: 14},
    16: {15: 13, 12: 9, 9: 14},
}

CROSSES = [0, 1, 2, 2, 5, 5, 4, 4, 1, 3, 4, 2, 3, 1, 3, 3, 5]

ROADS = [0, 33, 28, 33, 28, 33, 28, 33, 28, 12, 12, 12, 12, 10, 10, 10, 10]

CROSSES2 = {
    1: [3, 11],
    2: [13, 8],
    3: [16, 5],
    4: [2, 11],
    5: [10, 7],
    6: [4, 16],
    7: [1, 13],
    8: [10, 6],
    9: [2, 3],
    10: [15, 9, 14],
    11: [15, 14, 12],
    12: [6, 7],
    13: [9, 12, 14],
    14: [4, 5],
    15: [1, 8],
    16: [9, 12, 15],
}


def get_point_sector(segments, p):
    """Возвращает номер сегмента дороги, в котором находится данная точка."""
    for i, s in enumerate(segments):
        if cv2.countNonZero(cv2.bitwise_and(s, p)):
            return i + 1
    return None


def check_special_path(ss, se, dx, dy):
    """Проверяет, находятся ли точки в одном и том же секторе при непустом искомом пути."""
    if ss != se:
        return False
    if abs(dx) > 20 and ss in (9, 10):
        return dx > 0
    if abs(dx) > 20 and ss in (11, 12):
        return dx < 0
    if abs(dy) > 20 and ss in (15, 16):
        return dy > 0
    if abs(dy) > 20 and ss in (13, 14):
        return dy < 0
    if ss in (5, 2):
        return (dx < 0 and abs(dx) > 15) or (dy > 0 and (abs(dy) > 15))
    if ss in (1, 6):
        return (dx > 0 and abs(dx) > 15) or (dy < 0 and (abs(dy) > 15))
    if ss in (8, 3):
        return (dx < 0 and abs(dx) > 15) or (dy < 0 and (abs(dy) > 15))
    if ss in (4, 7):
        return (dx > 0 and abs(dx) > 15) or (dy > 0 and (abs(dy) > 15))
    return False


def load_segments():
    segments = []
    for s in range(1, 17):
        segments.append(
            cv2.inRange(
                cv2.imread(f"segments/s{s}.png"),
                (239, 255, 0)[::-1],
                (239, 255, 0)[::-1],
            )
        )
    return segments


def dijkstra(s, e):
    """Реализация алгоритма Дейкстры через PriorityQueue."""
    pq = PriorityQueue()

    dist = [10**9] * (16 + 1)
    prev = [-1] * (16 + 1)

    pq.put((0, s))
    dist[s] = 0

    while not pq.empty():
        u = pq.get()[1]

        for v, w in G[u].items():
            if dist[v] > dist[u] + w + ROADS[u]:
                dist[v] = dist[u] + w + ROADS[u]
                prev[v] = u
                pq.put((dist[v], v))

    path = []
    cur = e
    while cur != s:
        path.append(CROSSES[cur])
        cur = prev[cur]
    path.reverse()

    return [dist[e], path]


def find_the_shortest_way(image) -> list:
    red_dot = cv2.inRange(image, (230, 0, 0)[::-1], (255, 30, 30)[::-1])
    blue_dot = cv2.inRange(image, (0, 0, 230)[::-1], (30, 30, 255)[::-1])

    segments = load_segments()

    start, end = get_point_sector(segments, blue_dot), get_point_sector(
        segments, red_dot
    )

    start_x, start_y, *_ = cv2.boundingRect(
        cv2.findContours(blue_dot, cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_NONE)[0][0]
    )
    end_x, end_y, *_ = cv2.boundingRect(
        cv2.findContours(red_dot, cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_NONE)[0][0]
    )

    if not check_special_path(start, end, end_x - start_x, end_y - start_y):
        return dijkstra(start, end)[1]

    starts = CROSSES2[start]
    res = [dijkstra(i, end) for i in starts]
    for i in range(0, len(res)):
        res[i][0] += G[start][starts[i]]
        res[i][1].insert(0, CROSSES[starts[i]])

    mn = (10**9, None)
    for i in res:
        if i[0] < mn[0]:
            mn = tuple(i)
    return mn[1]
