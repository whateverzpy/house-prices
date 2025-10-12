# 房价预测 - Kaggle竞赛项目

本项目参与了 Kaggle 的 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 竞赛，目标是通过回归模型预测美国 Ames 城市住宅的最终售价。

## 项目结构

```txt
├── data/                # 数据集及提交结果
│   ├── train.csv        # 训练数据
│   ├── test.csv         # 测试数据
│   ├── sample_submission.csv  # 官方提交样例
│   ├── submission.csv   # 本项目最终提交结果
│   └── submission_xxx.csv  # 各模型单独预测结果
├── src/                 # 源代码
│   ├── main.py          # 主程序入口
│   ├── DataPreprocessor.py  # 数据预处理与特征工程
│   └── ModelTrainer.py  # 模型训练与融合
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明
```

## 环境依赖

请确保已安装以下依赖（详见 `requirements.txt`）：

- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm

安装方法：

```sh
pip install -r requirements.txt
```

## 主要流程

1. **数据预处理**  
   使用 `DataPreprocessor` 完成缺失值处理、特征工程（如总面积、总浴室数、房屋年龄等）、特征编码。

2. **模型训练与融合**  
   使用 `ModelTrainer` 训练多种回归模型（Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM），并进行交叉验证和模型融合。

3. **结果生成与提交**  
   在 `main.py` 中，融合模型预测结果并生成提交文件，保存于 `submission.csv`。

## 运行方法

在 src 目录下运行主程序：

```sh
python main.py
```

运行后会自动完成数据处理、模型训练、融合预测，并生成提交文件。

## 参考

- [竞赛主页](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
