import pandas as pd
import folium  # 交互式地图库

# ---------------------- 1. 读取经纬度数据 ----------------------
# 读取你的站点经纬度CSV（确保列名包含：站点名、经度、纬度，可根据实际调整）
# 若你的列名不同（如“station_name”“lon”“lat”），修改下面的列名即可
df = pd.read_csv("站点经纬度信息.csv")

# 查看数据结构（运行后会显示前5行，确认列名是否正确）
print("数据预览：")
print(df.head())
print(f"\n共 {len(df)} 个站点")

# ---------------------- 2. 定义地图基础参数 ----------------------
# 计算经纬度平均值，让地图默认居中显示所有站点
center_lat = df["纬度"].mean()  # 替换为你的纬度列名（如“lat”）
center_lon = df["经度"].mean()  # 替换为你的经度列名（如“lon”）

# 创建地图（zoom_start=10 表示默认缩放级别，数字越大越近）
m = folium.Map(
    location=[center_lat, center_lon],  # 地图中心点
    zoom_start=10,                     # 默认缩放级别
    tiles="OpenStreetMap"              # 地图样式（开源街道地图）
)

# ---------------------- 3. 批量标记所有站点 ----------------------
for index, row in df.iterrows():
    # 获取单个站点的信息（替换为你的实际列名）
    station_name = row["站点ID"]  # 站点名称（鼠标悬浮时显示）
    lat = row["纬度"]             # 纬度
    lon = row["经度"]             # 经度

    # 在地图上添加标记（红色圆点，大小10，鼠标悬浮显示站点名）
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,                  # 标记大小（像素）
        color="red",                # 边框颜色
        fill=True,
        fill_color="red",           # 填充颜色
        fill_opacity=0.7,           # 填充透明度
        # tooltip=station_name        # 鼠标悬浮显示的文字
    ).add_to(m)

     # ✅ 直接永久显示站点名称（不悬停）
    folium.map.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            icon_size=(150, 36),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">{station_name}</div>'
        )
    ).add_to(m)

# ---------------------- 4. 保存地图文件 ----------------------
# 保存为HTML，双击即可在浏览器打开
map_save_path = "湖南高速站点分布图-直接.html"
m.save(map_save_path)

print(f"\n✅ 地图已保存到：{map_save_path}")
print("✅ 操作提示：双击HTML文件，用浏览器打开即可查看交互式地图")
# FaST-main\Test\显示站点.py