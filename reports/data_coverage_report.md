# 数据覆盖报告

| dataset                | coverage_status     |   rows | start_date   | end_date   | notes                                                        |
|:-----------------------|:--------------------|-------:|:-------------|:-----------|:-------------------------------------------------------------|
| 300316 stock daily qfq | available           |   2044 | 2018-01-02   | 2026-06-09 | primary OHLCV data                                           |
| index template         | missing_or_template |      0 |              |            | manual index CSV format                                      |
| industry template      | missing_or_template |      0 |              |            | manual industry CSV format                                   |
| financial template     | missing_or_template |      0 |              |            | manual financial CSV format; ann_date required for model use |
| event template         | missing_or_template |      0 |              |            | manual event CSV format                                      |
| peer template          | missing_or_template |      0 |              |            | manual peer CSV format                                       |
| processed features     | available           |   2044 | 2018-01-02   | 2026-06-09 |                                                              |
| processed labels       | available           |   2044 | 2018-01-02   | 2026-06-09 |                                                              |

当前运行以本地 CSV 可复现为优先，未强制联网下载。`src/data_downloader.py` 已提供 Eastmoney 尝试下载接口；AKShare/Tushare 在当前环境缺失。指数、行业、财务、事件和 peer 数据当前只有模板或缺失，因此不会进入模型。后续补数必须保留来源、下载时间和公告日。
