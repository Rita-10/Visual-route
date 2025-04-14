# first line: 77
    @staticmethod
    @memory.cache
    def generate_path(start: tuple, end: tuple) -> list:
        """
        混合路径规划入口
        :param start: 起点坐标 (lat, lng)
        :param end: 终点坐标 (lat, lng)
        :return: 路径点列表 [{'lat':..., 'lng':...}, ...]
        """
        try:
            # 优先使用A*算法
            path = RoutePlanner._a_star_search(start, end)
            if not path:
                # 降级使用贝塞尔曲线
                path = RoutePlanner._bezier_fallback(start, end)
            return path
        except Exception as e:
            logging.error(f"路径规划失败: {e}")
            return []
