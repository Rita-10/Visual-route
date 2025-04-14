import logging
import time
from datetime import datetime
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
from flask import Flask, jsonify, render_template
from shapely.geometry import Point, LineString, MultiPolygon
from rtree import index
from waypoints_db import waypoints_db

# 调用航点数据库
#def get_nautical_waypoints(origin, dest):
    #return waypoints_db.get((origin, dest), [])


def process_data():
    # 在开头添加
    start_time = time.time()
    print(f"[{datetime.now()}] 开始处理数据...")
    # 数据加载与清洗
    df = pd.read_csv("cleaned_data.csv")

    # 重命名列 import 为 import_dest（避免关键字冲突）。过滤掉 lbs（货物重量）为 0 或负值的无效记录。
    df = df.rename(columns={"import": "import_dest"})
    df = df[df['lbs'] > 0]

    # 地理修正规则
    geo_corrections = {
        "United Kingdom": {"longitude2": -3.436},
        "Bombay": {"import_dest": "Mumbai", "latitude2": 19.0760, "longitude2": 72.8777},
        "Saigon": {"import_dest": "Ho Chi Minh City", "latitude2": 10.8231, "longitude2": 106.6297},
        "Singapore": {"latitude2": 1.9157, "longitude2": 104.1064},
        "Australia": {"latitude2": -26.8798, "longitude2": 153.0834},
        "Canada": {"latitude2": 51.6335, "longitude2": -128.49},
        "Monte Video": {"latitude2": -34.8574, "longitude2": -56.1511},
        "Sayam": {"latitude2": 13.5004, "longitude2": 100.4713}
    }

    # 应用修正
    for origin, correction in geo_corrections.items():
        print(f"修正 {origin} 地理数据，correction: {correction}")
        mask = df["import_dest"].str.contains(origin, case=False)
        df.loc[mask, list(correction.keys())] = list(correction.values())

    # 路径生成核心逻辑
    def generate_safe_route(row):
        waypoints = get_nautical_waypoints(row['export'], row['import_dest'])
        all_points = [
            (row['latitude1'], row['longitude1']),
            *waypoints,
            (row['latitude2'], row['longitude2'])
        ]

        path = []
        land_points = 0
app = Flask(__name__)


# 预加载陆地数据及空间索引
world_land = None
LAND_SINDEX = None
LAND_DATA = None

try:
    # 加载高精度海岸线数据
    land_shapefile_path = '110m_physical/ne_110m_land.shp'
    world_land = gpd.read_file(land_shapefile_path).buffer(0)

    # 坐标系处理增加容差参数
    if world_land.crs != 'EPSG:4326':
        world_land = world_land.to_crs('EPSG:4326', use_arrow=False)

    # 使用unary_union替代union_all
    LAND_DATA = world_land.geometry.union_all().buffer(0.1)
    print(f"成功加载陆地数据，覆盖区域面积 {LAND_DATA.area} 平方公里")
    # 海洋数据处理
    ocean_shapefile_path = '110m_physical/ne_110m_ocean.shp'
    world_ocean = gpd.read_file(ocean_shapefile_path).buffer(0)
    if world_ocean.crs != 'EPSG:4326':
        world_ocean = world_ocean.to_crs('EPSG:4326', use_arrow=False)

    OCEAN_DATA = world_ocean.geometry.union_all()
    print(f"海洋数据加载完成，包含 {len(world_ocean)} 个多边形")

    # 构建陆地空间索引
    LAND_SINDEX = index.Index()
    for idx, geom in enumerate(world_land.geometry):
        LAND_SINDEX.insert(idx, geom.bounds)
    # 构建海洋空间索引
    OCEAN_SINDEX = index.Index()
    for idx, geom in enumerate(world_ocean.geometry):
        OCEAN_SINDEX.insert(idx, geom.bounds)

except Exception as e:
    print(f"地理数据初始化失败: {str(e)}")
    raise  # 抛出异常避免后续逻辑出错

OCEAN_BUFFER = OCEAN_DATA.buffer(0.5)  # 新增海洋缓冲区
OCEAN_BUFFER_SINDEX = index.Index()  # 新增海洋缓冲区空间索引
for idx, geom in enumerate(OCEAN_BUFFER.geoms if isinstance(OCEAN_BUFFER, MultiPolygon) else [OCEAN_BUFFER]):
    OCEAN_BUFFER_SINDEX.insert(idx, geom.bounds)


# 缓存陆地检查函数
@lru_cache(maxsize=100000)
def cached_land_check(lng: float, lat: float) -> bool:
    """带缓存的地理位置检查"""
    point = Point(round(lng, 6), round(lat, 6))

    if LAND_SINDEX and not world_land.empty:
        possible = list(LAND_SINDEX.intersection(point.bounds))
        return any(world_land.geometry[i].contains(point) for i in possible)
    return LAND_DATA.contains(point)


@lru_cache(maxsize=100000)
def cached_ocean_check(lng: float, lat: float) -> bool:
    """带缓存的海洋缓冲区检查"""
    point = Point(round(lng, 6), round(lat, 6))
    if OCEAN_BUFFER_SINDEX and not OCEAN_BUFFER.is_empty:
        possible = list(OCEAN_BUFFER_SINDEX.intersection(point.bounds))
        return any(OCEAN_BUFFER.geoms[i].contains(point) for i in possible)
    return OCEAN_BUFFER.contains(point)


###未加入RRT之前3-4s处理完成，加入RRT后40s+（max_iter=500），根据情况减少迭代轮数。注释时avoid_land_crossing也一并替换########################################################################################
def rrt_planner(start, end, max_iter=0, step_size=0.5, goal_bias=0.3):
    """改进版RRT路径规划算法"""
    tree = [{'point': start, 'parent': None}]
    min_goal_dist = 0.5  # 约55公里

    # 动态采样区域（基于海洋缓冲区）
    """LineString([start, end]).length：计算起点start和终点end之间的直线距离（单位可能是经度/纬度或具体距离单位，取决于坐标系）。
    * 0.3：取该距离的30%作为基础缓冲距离（例如两点相距10单位，则缓冲距离为3单位）。
    max(3.0, ...)：确保缓冲距离至少为3.0（避免两点距离过近时搜索区域太小）。"""
    search_margin = max(3.0, LineString([start, end]).length * 0.3)
    " 以起终点为基准，向四周扩展search_margin的距离，形成一个矩形搜索区域。"
    min_lng = min(start[0], end[0]) - search_margin
    max_lng = max(start[0], end[0]) + search_margin
    min_lat = min(start[1], end[1]) - search_margin
    max_lat = max(start[1], end[1]) + search_margin

    """海洋优先（70%）： 提高采样效率，避免在陆地区域浪费计算资源（因为船舶路径必须在海洋中）。
    全局随机（30%）： 避免陷入局部最优，确保算法能探索动态区域内的所有可能性（例如绕过复杂障碍）。"""
    for _ in range(max_iter):
        # 海洋优先采样策略
        if np.random.random() < 0.7:  # 70%概率在海洋缓冲区采样
            target = ocean_biased_sampling(end, goal_bias)
        else:
            " 在范围 [a, b] 内生成均匀分布的随机浮点数。"
            target = (np.random.uniform(min_lng, max_lng),
                      np.random.uniform(min_lat, max_lat))

        """从树中找到最近的节点并向目标方向扩展新节点。
        功能： 在当前的树结构 tree 中，找到离随机采样点 target 最近的节点。
        np.hypot： 计算两点之间的欧氏距离（直线距离）
        min(tree, key=...)： 通过比较所有节点到 target 的距离，选择距离最小的节点作为 nearest。
        """
        nearest = min(tree, key=lambda n: np.hypot(n['point'][0] - target[0], n['point'][1] - target[1]))

        """
        功能：计算方向向量
        dx, dy： 表示从最近节点 nearest 指向目标 target 的方向向量。
        dist： 重新计算距离（与之前一致，用于后续归一化）。"""
        dx = target[0] - nearest['point'][0]
        dy = target[1] - nearest['point'][1]
        dist = np.hypot(dx, dy)

        """
        功能：距离过近时的处理
        作用： 如果最近节点与目标点几乎重合，跳过本次迭代。
        原因： 避免除以零错误（后续需要 dx/dist 和 dy/dist），同时节省计算资源。
        """
        if dist < 1e-6: continue

        """
        动态调整的控制点算法
        归一化方向向量： dx/dist 和 dy/dist 将方向向量转换为单位向量（长度为1）。
        步长控制： step_size 决定每次扩展的固定距离（例如0.5单位），平衡探索速度与精度。
        """
        new_point = (
            nearest['point'][0] + dx / dist * step_size,
            nearest['point'][1] + dy / dist * step_size
        )

        # 双重校验：路径不跨陆地且新点在海洋缓冲区
        if (not line_crosses_land(nearest['point'], new_point) and
                cached_ocean_check(new_point[0], new_point[1])):

            new_node = {'point': new_point, 'parent': nearest}
            tree.append(new_node)

            # 动态目标检查
            """np.hypot：计算新生成的点(new_point)和目标点(end)之间的欧几里得距离
            min_goal_dist：预设的终止阈值（如0.5单位，约55公里）
            作用：当新点距离终点足够近时，认为已经"到达"目标
            """
            if np.hypot(new_point[0] - end[0], new_point[1] - end[1]) < min_goal_dist:
                return path_simplify(reconstruct_path(new_node, end))

    # 后备策略：返回缓冲区内的最近路径
    return get_fallback_path(tree, end)


def ocean_biased_sampling(goal, goal_bias):
    """海洋优先采样策略"""
    if np.random.random() < goal_bias:
        return goal
    # 在海洋缓冲区范围内随机采样
    bounds = OCEAN_BUFFER.bounds
    return (
        np.random.uniform(bounds[0], bounds[2]),
        np.random.uniform(bounds[1], bounds[3])
    )


def path_simplify(raw_path):
    """路径简化：保留关键转折点
    range(0, len(raw_path), 2)： 生成一个从 0 开始、到路径长度结束、步长为 2 的索引序列（如 [0, 2, 4,...]）。
    raw_path[i] for i in ...： 通过列表推导式，选取索引对应的路径点（每隔一个点取一个）。
    + [raw_path[-1]]： 强制包含路径的最后一个点（终点），避免因步长间隔丢失终点。
    """
    return [raw_path[i] for i in range(0, len(raw_path), 2)] + [raw_path[-1]]


def get_fallback_path(tree, goal):
    """后备路径生成：选择缓冲区内的最近点
    列表推导式：遍历树中所有节点 n
    cached_ocean_check(x,y)：检查节点坐标是否在海洋中（可能是基于缓存的地理数据）
    结果：生成仅包含海洋节点的 valid_nodes 列表
    """
    valid_nodes = [n for n in tree if cached_ocean_check(n['point'][0], n['point'][1])]

    # 如果没有有效海洋节点，直接返回 None（表示规划失败）
    if not valid_nodes: return None

    # 选择距离目标最近的海洋节点
    nearest_to_goal = min(valid_nodes, key=lambda n: np.hypot(n['point'][0] - goal[0], n['point'][1] - goal[1]))
    return reconstruct_path(nearest_to_goal, goal)


def line_crosses_land(p1, p2):
    """检查线段是否与陆地相交"""
    line = LineString([p1, p2])
    return line.intersects(LAND_DATA)


def reconstruct_path(node, goal):
    """从目标节点回溯路径"""
    path = []
    while node is not None:
        path.append(node['point'])
        node = node['parent']
    path.reverse()
    path.append(goal)
    return path



# 贝塞尔曲线生成工具
def cubic_bezier(p0, p1, p2, p3, t, sea_status):
    """动态调整的控制点算法
    逻辑：如果 sea_status 中至少有3个点是海洋（sum(sea_status) >= 3），则使用较大的偏移量 0.002；否则使用较小的 0.0005。
    目的：在海洋区域较多时，生成更明显的曲线绕行（避免靠近潜在陆地）；否则生成轻微曲线。"""
    offset = 0.002 if sum(sea_status) >= 3 else 0.0005


    """
    B₀(t) = (1-t)³      # 起点 p0 的权重
    B₁(t) = 3(1-t)²t    # 第一控制点 p1 的权重
    B₂(t) = 3(1-t)t²    # 第二控制点 p2 的权重
    B₃(t) = t³          # 终点 p3 的权重
    """
    return (
            (1 - t) ** 3 * p0 +  # 起点权重
            3 * (1 - t) ** 2 * t * (p1 + offset) +   # 第一控制点（+偏移）
            3 * (1 - t) * t ** 2 * (p2 - offset) +   # 第二控制点（-偏移）
            t ** 3 * p3   # 终点权重
    )


def generate_bezier_segment(start, end, check_interval=0.01):
    """生成带实时校验的贝塞尔曲线段"""
    path = []
    lat1, lon1 = start
    lat2, lon2 = end

    # 检查陆地状态
    sea_status = [
        not cached_land_check(lon1, lat1),
        not cached_land_check((lon1 + lon2) / 2, (lat1 + lat2) / 2)
    ]

    t = 0
    while t <= 1:
        lat = cubic_bezier(lat1, lat1, lat2, lat2, t, sea_status)
        lng = cubic_bezier(lon1, lon1, lon2, lon2, t, sea_status)

        path.append({"lat": round(lat, 6), "lng": round(lng, 6)})
        t += check_interval

    return path


# 陆地规避算法无RRT##########################################
# def avoid_land_crossing(segment):
#     """使用航段检测和航点插入的规避算法"""
#     safe_path = []
#     prev_point = None
#
#     for point in segment:
#         if prev_point:
#             line = LineString([
#                 (prev_point['lng'], prev_point['lat']),
#                 (point['lng'], point['lat'])
#             ])
#
#             if line.intersects(LAND_DATA):
#                 # 插入绕行点
#                 bypass = generate_bypass_points(prev_point, point)
#                 safe_path.extend(bypass)
#             else:
#                 safe_path.append(point)
#         else:
#             safe_path.append(point)
#         prev_point = point
#
#     return safe_path

##加入RRT算法规避陆地，保底返回贝塞尔曲线段
def avoid_land_crossing(segment):
    print('调用RRT算法')

    safe_path = []
    prev_point = None
    print(f'1: {prev_point}')
    for point in segment:
        if prev_point:
            start = (prev_point['lng'], prev_point['lat'])
            end_pt = (point['lng'], point['lat'])

            ###LineString将两点转化为几何线段，用以检查两点之间连线是否与陆地相交，intersects返回布尔值是否相交
            if LineString([start, end_pt]).intersects(LAND_DATA):
                # 尝试RRT路径规划
                rrt_path = rrt_planner(start, end_pt)
                if rrt_path:
                    # 转换坐标格式并加入路径
                    safe_path.extend([{'lat': p[1], 'lng': p[0]} for p in rrt_path])
                    print('RRT算法成功')
                else:
                    # RRT失败时返回原始贝塞尔曲线段
                    print(f'prev_point: {prev_point}')
                    # bypass = generate_bypass_points(prev_point, point)
                    # safe_path.extend(bypass)
                    safe_path.append(point)
                    print('RRT算法失败，保底选择点位')
            else:
                safe_path.append(point)
        else:
            safe_path.append(point)
        prev_point = point
    return safe_path


# def generate_bypass_points(p1, p2):
#     """生成绕行三点"""
#     mid_lat = (p1['lat'] + p2['lat']) / 2 + 0.5
#     mid_lng = (p1['lng'] + p2['lng']) / 2 + 0.5
#
#     return [
#         p1,
#         {"lat": mid_lat, "lng": mid_lng},
#         p2
#     ]

# 调用航点数据库
def get_nautical_waypoints(origin, dest):
    return waypoints_db.get((origin, dest), [])


def process_data():
    # 在开头添加
    start_time = time.time()
    print(f"[{datetime.now()}] 开始处理数据...")
    # 数据加载与清洗
    df = pd.read_csv("cleaned_data.csv")

    # 重命名列 import 为 import_dest（避免关键字冲突）。过滤掉 lbs（货物重量）为 0 或负值的无效记录。
    df = df.rename(columns={"import": "import_dest"})
    df = df[df['lbs'] > 0]

    # 地理修正规则
    geo_corrections = {
        "United Kingdom": {"longitude2": -3.436},
        "Bombay": {"import_dest": "Mumbai", "latitude2": 19.0760, "longitude2": 72.8777},
        "Saigon": {"import_dest": "Ho Chi Minh City", "latitude2": 10.8231, "longitude2": 106.6297},
        "Singapore": {"latitude2": 1.9157, "longitude2": 104.1064},
        "Australia": {"latitude2": -26.8798, "longitude2": 153.0834},
        "Canada": {"latitude2": 51.6335, "longitude2": -128.49},
        "Monte Video": {"latitude2": -34.8574, "longitude2": -56.1511},
        "Sayam": {"latitude2": 13.5004, "longitude2": 100.4713}
    }

    # 应用修正
    for origin, correction in geo_corrections.items():
        print(f"修正 {origin} 地理数据，correction: {correction}")
        mask = df["import_dest"].str.contains(origin, case=False)
        df.loc[mask, list(correction.keys())] = list(correction.values())

    # 路径生成核心逻辑
    def generate_safe_route(row):
        waypoints = get_nautical_waypoints(row['export'], row['import_dest'])
        all_points = [
            (row['latitude1'], row['longitude1']),
            *waypoints,
            (row['latitude2'], row['longitude2'])
        ]

        path = []
        land_points = 0

        # 分段生成贝塞尔曲线
        for i in range(len(all_points) - 1):
            segment = generate_bezier_segment(
                all_points[i],
                all_points[i + 1],
                check_interval=0.1
            )

            # 陆地穿越检查与修正
            safe_segment = avoid_land_crossing(segment)
            # 记录被移除的陆地点数量 land_points
            land_points += len(segment) - len(safe_segment)
            path.extend(safe_segment)

        # 验证海上比例
        if len(path) == 0 or land_points / len(path) > 0.05:
            return []

        return path

    from tqdm import tqdm
    tqdm.pandas(desc="生成航线")
    df["path"] = df.apply(generate_safe_route, axis=1)
    print(f"[{datetime.now()}] 处理完成，耗时：{time.time() - start_time:.2f}s")
    return df[df['path'].apply(len) > 0].to_dict(orient="records")


# Flask路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    try:
        return jsonify({
            "routes": process_data(),
            "metadata": {
                "crs": "EPSG:4326",
                "pathResolution": 50
            }
        })
    except Exception as e:
        logging.error(f"数据生成失败: {str(e)}")
        print(f"数据生成失败: {str(e)}")
        return jsonify({"error": "航线生成服务暂时不可用"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
