import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, AvgPool1D
from tensorflow.keras.utils import plot_model


# ==================== 定义Net1模型（从原代码复制） ====================

class Net1(Model):
    """模型1：基础CNN模型"""

    def __init__(self, input_dim=205):
        super(Net1, self).__init__()
        self.conv1 = Conv1D(16, 3, padding='same', activation='relu', input_shape=(input_dim, 1))
        self.conv2 = Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv1D(64, 3, dilation_rate=2, padding='same', activation='relu')
        self.conv4 = Conv1D(64, 5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool1 = MaxPool1D(3, strides=2, padding='same')
        self.conv5 = Conv1D(128, 5, dilation_rate=2, padding='same', activation='relu')
        self.conv6 = Conv1D(128, 5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool2 = MaxPool1D(3, strides=2, padding='same')
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc21 = Dense(16, activation='relu')
        self.fc22 = Dense(256, activation='sigmoid')
        self.fc3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        return self.fc3(x1 + x2)


# ==================== 主程序 ====================

def main():
    print("=" * 50)
    print("生成 Net1 模型结构图")
    print("=" * 50)

    # 设置特征维度（与您的数据一致）
    FEATURE_DIM = 205

    # 1. 创建模型
    print(f"\n创建 Net1 模型 (输入维度: {FEATURE_DIM})...")
    model = Net1(input_dim=FEATURE_DIM)

    # 2. 构建模型（必须用虚拟数据过一遍，否则plot_model无法获取形状）
    print("构建模型...")
    dummy_input = tf.random.normal((1, FEATURE_DIM, 1))
    _ = model(dummy_input)

    # 3. 打印模型结构（文本）
    print("\n模型结构 (summary):")
    model.summary()

    # 4. 保存模型结构图（图片）
    print("\n生成模型结构图...")
    try:
        plot_model(
            model,
            to_file='Net1_model_structure.png',
            show_shapes=True,  # 显示每层输出形状
            show_layer_names=True,  # 显示层名称
            dpi=300,  # 高清分辨率
            expand_nested=True  # 展开嵌套层
        )
        print("✓ 模型结构图已保存: Net1_model_structure.png")
        print("  文件位置: ./Net1_model_structure.png")
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        print("\n可能的解决方案:")
        print("  1. 安装 pydot: pip install pydot")
        print("  2. 安装 graphviz: pip install graphviz")
        print("  3. Windows用户还需要安装Graphviz软件:")
        print("     https://graphviz.org/download/")
        print("     并添加到系统PATH环境变量")
        print("\n替代方案: 使用上面打印的 summary() 文本截图即可")

    print("\n" + "=" * 50)
    print("完成！")


if __name__ == "__main__":
    main()