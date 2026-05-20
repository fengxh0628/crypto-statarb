import datetime
from binance_historical_data import BinanceDataDumper

# 1. 配置下载参数
# path_dir_where_to_dump: 设置数据保存的文件夹，可以换成你的路径，比如 './my_binance_data'
# asset_class: 'um' 代表U本位永续合约 (USDⓈ-M Futures)
# data_type: 'klines' 代表K线数据
# data_frequency: '5m' 就是我们需要的5分钟粒度
data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="./binance_data", 
    asset_class="um",            
    data_type="klines",          
    data_frequency="5m",         
)

# 2. 执行下载 (核心操作！)
# 下面的命令会下载所有U本位永续合约，从最早有数据开始到今天的所有5分钟K线
print("开始下载全量数据... (首次下载约40分钟)")
data_dumper.dump_data()

print("全量数据下载完成！")

# 3. (可选) 如何更新数据
# 未来当你需要获取最新的数据时，再次运行 data_dumper.dump_data() 即可。
# 程序会自动检查已下载的数据，只下载新产生的部分，速度会快很多。
# print("开始增量更新...")
# data_dumper.dump_data()
# print("增量更新完成！")

# 4. (可选) 如何下载指定时间范围的数据
# 如果你只想下载某个时间段的数据，可以取消下面代码的注释来使用
# data_dumper.dump_data(
#     date_start=datetime.date(year=2024, year=1, day=1),
#     date_end=datetime.date(year=2024, year=12, day=31),
# )
