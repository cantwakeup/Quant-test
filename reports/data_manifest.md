# 数据清单

生成时间：2026-06-10T20:57:06

| dataset                | exists   |   rows | start_date   | end_date   | source    | notes                                                        |
|:-----------------------|:---------|-------:|:-------------|:-----------|:----------|:-------------------------------------------------------------|
| 300316 stock daily qfq | True     |   2044 | 2018-01-02   | 2026-06-09 | local_csv | primary OHLCV data                                           |
| index template         | True     |      0 |              |            | template  | manual index CSV format                                      |
| industry template      | True     |      0 |              |            | template  | manual industry CSV format                                   |
| financial template     | True     |      0 |              |            | template  | manual financial CSV format; ann_date required for model use |
| event template         | True     |      0 |              |            | template  | manual event CSV format                                      |
| peer template          | True     |      0 |              |            | template  | manual peer CSV format                                       |
| processed features     | True     |   2044 | 2018-01-02   | 2026-06-09 | pipeline  |                                                              |
| processed labels       | True     |   2044 | 2018-01-02   | 2026-06-09 | pipeline  |                                                              |

缺失数据不会被伪造。模板文件只定义人工补数格式，不会进入模型特征。
