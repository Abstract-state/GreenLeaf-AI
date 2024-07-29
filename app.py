from flask import Flask, request, render_template
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Define the convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Define the ResNet9 model architecture
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super(ResNet9, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)

        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Instantiate and load your model
model_path = 'plant-disease-model.pth'
model = ResNet9(3, 38)  # Assuming 3 input channels (RGB) and 38 classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define your image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class_names = [
    "Tree = Tree, Condition: Apple scab \n, Treatment: Apply fungicides and practice proper orchard hygiene to control the spread.",
    "Tree = Apple, Condition: Black rot \n, Treatment: Use fungicides, prune infected parts, and manage orchard sanitation.",
    "Tree = Apple, Condition: Cedar apple rust \n Treatment: Apply fungicides, remove nearby cedar trees, and practice proper orchard management.",
    "Tree = Apple, Condition: Healthy",
    "Tree = Blueberry, Condition: Healthy",

    "Tree = Cherry (including sour), Condition: Healthy",
    "Tree = Cherry (including sour), Condition: Powdery mildew \n Use fungicides and maintain good air circulation in the orchard.",
    "Tree = Corn (maize), Condition: Cercospora leaf spot, Gray leaf spot \n Treatment : Control insect vectors, use resistant varieties, and practice proper sanitation.",
    "Tree = Corn (maize), Condition: Common rust \n Treatment: Plant resistant varieties and consider fungicide treatments in severe cases.",
    "Tree = Corn (maize), Condition: Healthy \n Treatment: Apply fungicides, remove nearby cedar trees, and practice proper orchard management.",
    "Tree = Corn (maize), Condition: Northern Leaf Blight ",
    "Tree = Grape, Condition: Black rot",
    "Tree = Grape, Condition: Esca (Black Measles)",
    "Tree = Grape, Condition: Healthy",
    "Tree = Grape, Condition: Leaf blight (Isariopsis Leaf Spot)",
    "Tree = Orange, Condition: Huanglongbing (Citrus greening)",
    "Tree = Peach, Condition: Bacterial spot",
    "Tree = Peach, Condition: Healthy",
    "Tree = Pepper, bell, Condition: Bacterial spot",
    "Tree = Pepper, bell, Condition: Healthy",
    "Tree = Potato, Condition: Early blight",
    "Tree = Potato, Condition: Healthy",
    "Tree = Potato, Condition: Late blight",
    "Tree = Raspberry, Condition: Healthy",
    "Tree = Soybean, Condition: Healthy",
    "Tree = Squash, Condition: Powdery mildew",
    "Tree = Strawberry, Condition: Healthy",
    "Tree = Strawberry, Condition: Leaf scorch",
    "Tree = Tomato, Condition: Bacterial spot \n Treatment : Use copper-based sprays and practice crop rotation",
    "Tree = Tomato, Condition: Early blight",
    "Tree = Tomato, Condition: Healthy",
    "Tree = Tomato, Condition: Late blight",
    "Tree = Tomato, Condition: Leaf Mold",
    "Tree = Tomato, Condition: Septoria leaf spot",
    "Tree = Tomato, Condition: Spider mites, Two-spotted spider mite",
    "Tree = Tomato, Condition: Target Spot",
    "Tree = Tomato, Condition: Tomato Yellow Leaf Curl Virus",
    "Tree = Tomato, Condition: Tomato mosaic virus"
]




@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join('static', image_file.filename)
            image_file.save(image_path)
            img = Image.open(image_path)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]
            return render_template('index.html', prediction=prediction, image_loc=image_file.filename)
    return render_template('index.html', prediction=None, image_loc=None)


if __name__ == '__main__':
    app.run(debug=True)
