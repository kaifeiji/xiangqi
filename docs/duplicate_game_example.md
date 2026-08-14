# 重复棋谱原始内容示例

去重键：`FEN + 完整 ICCS 着法序列 + Result`

两局来源均为 `data/raw/dpxq-99813games.pgns`，局号分别为 2874 和 2876。双方信息不同，但原始对局内容完全相同，因此第二局不会再次导出训练样本。

## 局 2874

```text
[Game "Chinese Chess"]
[Event "2013年第五届东坡杯中国象棋公开赛"]
[Site "-"]
[Date "2013-08-16"]
[Round "-"]
[RedTeam "湖北武汉"]
[Red "湖北武汉 刘宗泽"]
[BlackTeam "四川眉山东坡区"]
[Black "四川眉山东坡区 夏睿"]
[Result "1-0"]
[Opening "-"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]
[Format "ICCS"]
G3-G4 H7-G7 B2-E2 G9-E7 F0-E1 A9-A8 H2-F2 H9-F8 H0-G2 I9-H9 B0-C2 A8-D8 A0-B0 D8-D4 G0-I2 B9-A7 C0-A2 A6-A5 C3-C4 D4-D5 G2-F4 D5-H5 B0-B6 F9-E8 I0-F0 G7-F7 F2-F7 B7-F7 F4-G6 H5-H6 G4-G5 E7-G5 E2-E6 F7-E7 E6-H6 F8-G6 H6-C6 H9-H2 C6-E6 H2-C2 E6-E4 G6-E5 B6-H6 E8-F9 F0-F5
```

## 局 2876

```text
[Game "Chinese Chess"]
[Event "2013年第五届东坡杯中国象棋公开赛"]
[Site "-"]
[Date "2013-08-16"]
[Round "-"]
[RedTeam "四川雅安"]
[Red "四川雅安 杨辉"]
[BlackTeam "四川大邑"]
[Black "四川大邑 赵修祥"]
[Result "1-0"]
[Opening "-"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]
[Format "ICCS"]
G3-G4 H7-G7 B2-E2 G9-E7 F0-E1 A9-A8 H2-F2 H9-F8 H0-G2 I9-H9 B0-C2 A8-D8 A0-B0 D8-D4 G0-I2 B9-A7 C0-A2 A6-A5 C3-C4 D4-D5 G2-F4 D5-H5 B0-B6 F9-E8 I0-F0 G7-F7 F2-F7 B7-F7 F4-G6 H5-H6 G4-G5 E7-G5 E2-E6 F7-E7 E6-H6 F8-G6 H6-C6 H9-H2 C6-E6 H2-C2 E6-E4 G6-E5 B6-H6 E8-F9 F0-F5
```

两局共同的 `game_id`：

```text
fb510e9f55dc40367348012cb4655306c97cdccf7c52d9738c8a37c8e04e4889
```
