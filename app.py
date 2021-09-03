import flask
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# CLASS_NAMES = os.listdir('data/dataset/train')
# CLASS_NAMES.pop(CLASS_NAMES.index('.DS_Store'))
# print(CLASS_NAMES)
CLASS_NAMES = ['Piet_Mondrian', 'Diego_Velazquez', 'Edgar_Degas', 'Georges_Seurat', 'Francisco_Goya', 'Andrei_Rublev',
               'Alfred_Sisley', 'Michelangelo', 'Rene_Magritte', 'Titian', 'Edouard_Manet', 'Giotto_di_Bondone',
               'Andy_Warhol', 'Jan_van_Eyck', 'El_Greco', 'Eugene_Delacroix', 'Pieter_Bruegel', 'Paul_Klee',
               'Paul_Gauguin', 'Claude_Monet', 'Marc_Chagall', 'Sandro_Botticelli', 'Henri_de_Toulouse-Lautrec',
               'Kazimir_Malevich', 'Paul_Cezanne', 'Camille_Pissarro', 'Salvador_Dali', 'Diego_Rivera',
               'Vasiliy_Kandinskiy', 'Gustav_Klimt', 'Vincent_van_Gogh', 'Gustave_Courbet', 'Amedeo_Modigliani',
               'Henri_Matisse', 'Frida_Kahlo', 'Pablo_Picasso', 'Jackson_Pollock', 'Pierre-Auguste_Renoir', 'Joan_Miro',
               'Peter_Paul_Rubens', 'Edvard_Munch', 'Caravaggio', 'Hieronymus_Bosch', 'Mikhail_Vrubel', 'Raphael',
               'Rembrandt', 'Leonardo_da_Vinci', 'Henri_Rousseau', 'William_Turner']
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
