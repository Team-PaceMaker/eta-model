from flask import Flask, request, jsonify
from torchvision.transforms import transforms

from PIL import Image

import torch

from vgg_model import VGG

app = Flask(__name__)

# 모델 로드
device = torch.device('cpu')

model = VGG()
model.load_state_dict(torch.load("model_state_dict_300_VGG.pt", map_location=device))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route('/attention', methods=['POST'])
def predict():
    # POST 요청에서 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    image_file = request.files['image']
    try:
        image = Image.open(image_file)
    except:
        return jsonify({'error': 'Invalid image file'}), 400

    image = transform(image)
    
    # 추론
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    prediction = output.argmax(dim=1).item()
    
    # 결과
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
