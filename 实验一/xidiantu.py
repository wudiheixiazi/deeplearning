# 构建完整的心电图分类模型训练代码（训练集有标签，测试集无标签）
def build_ecg_classification_system():
    """
    构建完整的心电图分类系统，包括数据预处理、模型定义、训练和评估
    训练集使用标签，测试集不使用标签
    """

    # ---------------------- 1. 数据预处理模块 ----------------------
    def preprocess_data(train_file_path, test_file_path):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # 读取训练数据（有标签）
        train_df = pd.read_csv(train_file_path)

        # 读取测试数据（无标签）
        test_df = pd.read_csv(test_file_path)

        # 解析心跳信号函数
        def parse_signal(signal_str):
            return np.array([float(x) for x in signal_str.split(',')])

        # 处理训练数据
        train_df['signal_array'] = train_df['heartbeat_signals'].apply(parse_signal)
        train_signal_matrix = np.vstack(train_df['signal_array'].values)
        signal_length = train_signal_matrix.shape[1]

        # 训练集特征和标签
        X_train = train_signal_matrix
        y_train = train_df['label'].astype(int)
        num_classes = len(np.unique(y_train))

        # 处理测试数据（只有信号，没有标签）
        test_df['signal_array'] = test_df['heartbeat_signals'].apply(parse_signal)
        X_test = np.vstack(test_df['signal_array'].values)

        # 标准化（只使用训练集拟合，然后转换训练集和测试集）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 对训练标签进行独热编码
        from tensorflow.keras.utils import to_categorical
        y_train_onehot = to_categorical(y_train, num_classes=num_classes)

        return (X_train_scaled, X_test_scaled, y_train_onehot, y_train,
                scaler, signal_length, num_classes)

    # ---------------------- 2. 模型定义模块 ----------------------
    def create_ecg_models(input_shape, num_classes):
        """创建两种模型：全连接神经网络和LSTM模型"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization

        # 模型1：全连接神经网络
        mlp_model = Sequential(name="ECG_MLP_Model")
        mlp_model.add(Dense(128, activation='relu', input_shape=input_shape))
        mlp_model.add(BatchNormalization())
        mlp_model.add(Dropout(0.3))
        mlp_model.add(Dense(64, activation='relu'))
        mlp_model.add(BatchNormalization())
        mlp_model.add(Dropout(0.3))
        mlp_model.add(Dense(32, activation='relu'))
        mlp_model.add(Dense(num_classes, activation='softmax'))

        # 编译模型
        mlp_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 模型2：LSTM模型
        lstm_input_shape = (input_shape[0], 1)
        lstm_model = Sequential(name="ECG_LSTM_Model")
        lstm_model.add(LSTM(64, return_sequences=True, input_shape=lstm_input_shape))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(LSTM(32, return_sequences=False))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(Dense(16, activation='relu'))
        lstm_model.add(Dense(num_classes, activation='softmax'))

        # 编译模型
        lstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return mlp_model, lstm_model

        # ---------------------- 3. 模型训练模块 ----------------------

    def train_model(model, X_train, y_train, X_val, y_val, model_type='mlp'):
        """训练模型并返回训练历史"""
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f'C:/Users/22314/Desktop/test/best_ecg_{model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        # 调整LSTM输入形状
        if model_type == 'lstm':
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], 1)

        # 从训练集中划分验证集（20%用于验证）
        from sklearn.model_selection import train_test_split
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # 训练模型
        history = model.fit(
            X_train_final, y_train_final,
            batch_size=32,
            epochs=50,
            validation_data=(X_val_final, y_val_final),
            callbacks=callbacks,
            verbose=1
        )

        return history, model

    # ---------------------- 4. 模型预测模块 ----------------------
    def predict_with_model(model, X_test, model_type='mlp'):
        """使用训练好的模型对测试集进行预测"""
        import numpy as np

        # 调整LSTM输入形状
        if model_type == 'lstm':
            X_test_input = X_test.reshape(-1, X_test.shape[1], 1)
        else:
            X_test_input = X_test

        # 预测
        y_pred = model.predict(X_test_input)
        y_pred_classes = np.argmax(y_pred, axis=1)

        return y_pred_classes

    # ---------------------- 5. 模型评估模块（仅验证集） ----------------------
    def evaluate_model(model, X_val, y_val, model_type='mlp'):
        """评估模型性能（使用验证集）"""
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 调整LSTM输入形状
        if model_type == 'lstm':
            X_val_input = X_val.reshape(-1, X_val.shape[1], 1)
        else:
            X_val_input = X_val

        # 预测
        y_pred = model.predict(X_val_input)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # 计算指标
        accuracy = accuracy_score(y_val, y_pred_classes)
        conf_matrix = confusion_matrix(y_val, y_pred_classes)
        class_report = classification_report(y_val, y_pred_classes)

        print(f"\n{model.name} 评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("\n分类报告:")
        print(class_report)

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[f'类别 {i}' for i in range(conf_matrix.shape[0])],
            yticklabels=[f'类别 {i}' for i in range(conf_matrix.shape[0])]
        )
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title(f'{model.name} 混淆矩阵')
        plt.savefig(f'C:/Users/22314/Desktop/test/{model.name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        return accuracy, conf_matrix, class_report

    # ---------------------- 6. 训练历史可视化模块 ----------------------
    def plot_training_history(history, model_name):
        """绘制训练损失和准确率曲线"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title(f'{model_name} 准确率曲线')
        plt.xlabel('epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{model_name} 损失曲线')
        plt.xlabel('epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'C:/Users/22314/Desktop/test/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------- 7. 保存预测结果模块 ----------------------
    def save_predictions(y_pred, output_path):
        """保存预测结果到CSV文件"""
        import pandas as pd

        results_df = pd.DataFrame({
            'predicted_label': y_pred
        })
        results_df.to_csv(output_path, index=False)
        print(f"\n预测结果已保存到: {output_path}")

    # ---------------------- 8. 主执行流程 ----------------------
    def main(train_file_path, test_file_path):
        import pandas as pd

        print("=" * 60)
        print("开始心电图分类模型训练流程")
        print("训练集（有标签）-> 测试集（无标签）")
        print("=" * 60)

        # 1. 数据预处理
        print("\n1. 正在进行数据预处理...")
        (X_train, X_test, y_train_onehot, y_train_raw,
         scaler, signal_length, num_classes) = preprocess_data(train_file_path, test_file_path)

        print(f"数据预处理完成:")
        print(f"- 信号长度: {signal_length}")
        print(f"- 类别数量: {num_classes}")
        print(f"- 训练集大小: {X_train.shape}")
        print(f"- 测试集大小: {X_test.shape}")

        # 2. 创建模型
        print("\n2. 正在创建模型...")
        input_shape = (signal_length,)
        mlp_model, lstm_model = create_ecg_models(input_shape, num_classes)

        print("\n全连接神经网络结构:")
        mlp_model.summary()
        print("\nLSTM模型结构:")
        lstm_model.summary()

        # 3. 训练模型（使用训练集，自动划分验证集）
        print("\n3. 正在训练全连接神经网络...")
        mlp_history, mlp_trained = train_model(
            mlp_model, X_train, y_train_onehot, X_train, y_train_onehot, model_type='mlp'
        )

        print("\n4. 正在训练LSTM模型...")
        lstm_history, lstm_trained = train_model(
            lstm_model, X_train, y_train_onehot, X_train, y_train_onehot, model_type='lstm'
        )

        # 4. 可视化训练历史
        print("\n5. 正在生成训练可视化图表...")
        plot_training_history(mlp_history, mlp_model.name)
        plot_training_history(lstm_history, lstm_model.name)

        # 5. 对测试集进行预测
        print("\n6. 正在对测试集进行预测...")
        mlp_predictions = predict_with_model(mlp_trained, X_test, model_type='mlp')
        lstm_predictions = predict_with_model(lstm_trained, X_test, model_type='lstm')

        # 6. 保存预测结果
        print("\n7. 正在保存预测结果...")
        save_predictions(mlp_predictions, 'C:/Users/22314/Desktop/test/test_predictions_mlp.csv')
        save_predictions(lstm_predictions, 'C:/Users/22314/Desktop/test/test_predictions_lstm.csv')

        # 7. 保存结果对比
        results_df = pd.DataFrame({
            '模型': ['全连接神经网络', 'LSTM模型'],
            '模型文件': ['best_ecg_mlp_model.h5', 'best_ecg_lstm_model.h5'],
            '训练历史图': ['ECG_MLP_Model_training_history.png', 'ECG_LSTM_Model_training_history.png'],
            '预测结果文件': ['test_predictions_mlp.csv', 'test_predictions_lstm.csv']
        })

        results_df.to_csv('C:/Users/22314/Desktop/test/model_training_results.csv', index=False, encoding='utf-8')

        print("\n" + "=" * 60)
        print("模型训练和预测完成！")
        print("=" * 60)
        print("\n结果汇总:")
        print(results_df.to_string(index=False))

        return {
            'mlp_model': mlp_trained,
            'lstm_model': lstm_trained,
            'scaler': scaler,
            'mlp_predictions': mlp_predictions,
            'lstm_predictions': lstm_predictions,
            'results': results_df
        }

    return main


# 创建训练系统并执行
ecg_training_system = build_ecg_classification_system()
training_results = ecg_training_system('C:/Users/22314/Desktop/test/train_set.csv',
                                       'C:/Users/22314/Desktop/test/test_set.csv')

print("\n" + "=" * 80)
print("最终结果")
print("=" * 80)
print("\n预测结果已生成，保存在 C:/Users/22314/Desktop/test/ 目录下:")
print("  - test_predictions_mlp.csv (全连接神经网络预测结果)")
print("  - test_predictions_lstm.csv (LSTM模型预测结果)")