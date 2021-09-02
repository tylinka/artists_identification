import flask
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = os.listdir('data/dataset/train')
CLASS_NAMES.pop(CLASS_NAMES.index('.DS_Store'))
NUM_CLASSES = len(CLASS_NAMES)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = flask.Flask(__name__)


def load_model(path):
    device = torch.device('cpu') # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnext50_32x4d(pretrained=True, progress=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_ft.eval()
    return model_ft


MODEL = load_model('models/resnext50_32x4d_gpu.pth')


def make_predictions(path_to_file, model):
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    img = Image.open(path_to_file)
    x = test_transform(img)
    x = x.unsqueeze(0)
    output = model(x)
    prediction = torch.argmax(output, 1)
    return CLASS_NAMES[prediction]


@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/predict', methods=["POST"])
def transform_view():
    f = flask.request.files.get('picture')
    if not f:
        return "No file"
    output = make_predictions(f, MODEL)
    response = flask.make_response(output)
    return flask.render_template("answer.html", output=output.split('_'))


if __name__ == "__main__":
    app.run(debug=False)
