import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from model import MNISTNet
import os

def load_model(model_path='mnist_model.pth'):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f'模型已从 {model_path} 加载')
        return model, device
    else:
        raise FileNotFoundError(f'模型文件 {model_path} 不存在，请先训练模型')

def preprocess_image(image_path):
    """预处理单张图片"""
    # 数据预处理，与训练时保持一致
    transform = transforms.Compose([
        transforms.Grayscale(),  # 确保是灰度图
        transforms.Resize((28, 28)),  # 调整大小到28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载并预处理图片
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path  # 假设传入的是PIL图像对象
    
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return image_tensor

def predict_single_image(model, device, image_tensor):
    """对单张图片进行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def infer_image(image_path, model_path='mnist_model.pth'):
    """推理单张图片的完整流程"""
    try:
        # 加载模型
        model, device = load_model(model_path)
        
        # 预处理图片
        image_tensor = preprocess_image(image_path)
        
        # 进行预测
        predicted_class, confidence, all_probs = predict_single_image(model, device, image_tensor)
        
        # 输出结果
        print(f'预测结果: {predicted_class}')
        print(f'置信度: {confidence:.4f}')
        print('所有类别的概率:')
        for i, prob in enumerate(all_probs):
            print(f'  类别 {i}: {prob:.4f}')
            
        return predicted_class, confidence
        
    except Exception as e:
        print(f'推理过程中出现错误: {e}')
        return None, None

def demo_with_mnist_test():
    """使用MNIST测试集中的图片进行演示"""
    from torchvision import datasets
    
    # 加载MNIST测试集
    test_dataset = datasets.MNIST(root='../data', train=False, download=True)
    
    # 随机选择一张图片
    idx = np.random.randint(0, len(test_dataset))
    image, true_label = test_dataset[idx]
    
    print(f'真实标签: {true_label}')
    
    # 进行推理
    predicted_class, confidence = predict_single_image_from_pil(image)
    
    return predicted_class, true_label, confidence

def predict_single_image_from_pil(pil_image, model_path='mnist_model.pth'):
    """从PIL图像对象进行预测"""
    try:
        model, device = load_model(model_path)
        image_tensor = preprocess_image(pil_image)
        predicted_class, confidence, all_probs = predict_single_image(model, device, image_tensor)
        
        print(f'预测结果: {predicted_class}')
        print(f'置信度: {confidence:.4f}')
        
        return predicted_class, confidence
    except Exception as e:
        print(f'推理过程中出现错误: {e}')
        return None, None

if __name__ == '__main__':
    # 使用示例
    print("MNIST手写数字识别推理演示")
    print("="*50)
    
    # 如果有具体的图片路径，可以这样使用：
    # image_path = 'path/to/your/image.png'
    # predicted_class, confidence = infer_image(image_path)
    
    # 使用MNIST测试集进行演示
    try:
        demo_with_mnist_test()
    except Exception as e:
        print(f'演示失败: {e}')
        print('请确保已经训练好模型，并且安装了所需的依赖包')