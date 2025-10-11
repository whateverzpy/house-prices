import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.numeric_features = []
        self.categorical_features = []

    def load_data(self, train_path, test_path):
        """加载训练和测试数据"""
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.train_len = len(self.train)

        # 保存目标变量
        self.y_train = np.log1p(self.train["SalePrice"])

        # 合并数据以便统一处理
        self.train = self.train.drop(["SalePrice"], axis=1)
        self.all_data = pd.concat([self.train, self.test], axis=0, ignore_index=True)

        return self.all_data

    def handle_missing_values(self):
        """处理缺失值"""
        # 对于某些特征，NA表示"无"
        none_cols = [
            "PoolQC",
            "MiscFeature",
            "Alley",
            "Fence",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "MasVnrType",
        ]

        for col in none_cols:
            if col in self.all_data.columns:
                self.all_data[col].fillna("None", inplace=True)

        # 数值型特征用0填充
        zero_cols = [
            "GarageYrBlt",
            "GarageArea",
            "GarageCars",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
        ]

        for col in zero_cols:
            if col in self.all_data.columns:
                self.all_data[col].fillna(0, inplace=True)

        # LotFrontage用中位数填充
        if "LotFrontage" in self.all_data.columns:
            self.all_data["LotFrontage"].fillna(
                self.all_data["LotFrontage"].median(), inplace=True
            )

        # 其他类别型特征用众数填充
        for col in self.all_data.columns:
            if self.all_data[col].dtype == "object":
                self.all_data[col].fillna(self.all_data[col].mode()[0], inplace=True)
            elif self.all_data[col].isnull().sum() > 0:
                self.all_data[col].fillna(self.all_data[col].median(), inplace=True)

        return self.all_data

    def feature_engineering(self):
        """特征工程"""
        # 总面积
        self.all_data["TotalSF"] = (
            self.all_data["TotalBsmtSF"]
            + self.all_data["1stFlrSF"]
            + self.all_data["2ndFlrSF"]
        )

        # 总浴室数
        self.all_data["TotalBath"] = (
            self.all_data["FullBath"]
            + 0.5 * self.all_data["HalfBath"]
            + self.all_data["BsmtFullBath"]
            + 0.5 * self.all_data["BsmtHalfBath"]
        )

        # 房屋年龄
        self.all_data["HouseAge"] = self.all_data["YrSold"] - self.all_data["YearBuilt"]
        self.all_data["RemodAge"] = (
            self.all_data["YrSold"] - self.all_data["YearRemodAdd"]
        )

        # 是否有车库
        self.all_data["HasGarage"] = self.all_data["GarageArea"].apply(
            lambda x: 1 if x > 0 else 0
        )

        # 是否有地下室
        self.all_data["HasBsmt"] = self.all_data["TotalBsmtSF"].apply(
            lambda x: 1 if x > 0 else 0
        )

        # 是否有第二层
        self.all_data["Has2ndFloor"] = self.all_data["2ndFlrSF"].apply(
            lambda x: 1 if x > 0 else 0
        )

        return self.all_data

    def encode_features(self):
        """编码分类特征"""
        # 识别数值型和分类型特征
        self.numeric_features = self.all_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_features = self.all_data.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # 对分类特征进行标签编码
        for col in self.categorical_features:
            le = LabelEncoder()
            self.all_data[col] = le.fit_transform(self.all_data[col].astype(str))
            self.label_encoders[col] = le

        return self.all_data

    def split_data(self):
        """分割训练集和测试集"""
        train = self.all_data[: self.train_len]
        test = self.all_data[self.train_len :]

        return train, test, self.y_train

    def process(self, train_path, test_path):
        """完整的预处理流程"""
        self.load_data(train_path, test_path)
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_features()

        return self.split_data()
