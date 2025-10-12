import pandas as pd
import numpy as np
from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
import warnings

warnings.filterwarnings("ignore")


def main():
    # 数据路径
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    print("=" * 50)
    print("Step 1: Data Preprocessing")
    print("=" * 50)

    # 数据预处理
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train = preprocessor.process(train_path, test_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target shape: {y_train.shape}")

    print("\n" + "=" * 50)
    print("Step 2: Model Training and Cross-Validation")
    print("=" * 50)

    # 模型训练
    trainer = ModelTrainer()
    trainer.init_models()

    # 交叉验证
    print("\nCross-validation scores:")
    cv_scores = trainer.cross_validate(X_train, y_train)

    # 训练模型
    print("\n" + "=" * 50)
    print("Step 3: Training Final Models")
    print("=" * 50)
    trainer.train_models(X_train, y_train)

    # 预测
    print("\n" + "=" * 50)
    print("Step 4: Making Predictions")
    print("=" * 50)
    trainer.predict(X_test)

    # 集成预测
    # ensemble_pred = trainer.ensemble_predict()
    ensemble_pred = trainer.weighted_ensemble()

    # 保存结果
    print("\n" + "=" * 50)
    print("Step 5: Saving Predictions")
    print("=" * 50)

    # 读取测试集ID
    test_ids = pd.read_csv(test_path)["Id"]

    # 创建提交文件
    submission = pd.DataFrame(
        {"Id": test_ids, "SalePrice": np.expm1(ensemble_pred)}  # 反对数变换
    )

    submission.to_csv("../data/submission.csv", index=False)
    print("Predictions saved to '../data/submission.csv'")

    # 也保存单个模型的预测结果以供参考
    for name, pred in trainer.predictions.items():
        individual_submission = pd.DataFrame(
            {"Id": test_ids, "SalePrice": np.expm1(pred)}
        )
        individual_submission.to_csv(f"../data/submission_{name}.csv", index=False)

    print("\nAll predictions saved successfully!")
    print("\nBest performing models based on CV:")
    sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1]["mean"])
    for i, (name, score) in enumerate(sorted_scores[:3], 1):
        print(f"{i}. {name}: {score['mean']:.4f}")


if __name__ == "__main__":
    main()
