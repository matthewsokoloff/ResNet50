import torch
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet50 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_resnet50',
    pretrained=True
)
utils = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_convnets_processing_utils'
)

resnet50.eval().to(device)

# imgs here
image_paths = [
    "images/beach-test-1.jpg"
]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# inference
images = []
for path in image_paths:
    img = Image.open(path).convert("RGB")
    images.append(preprocess(img))

batch = torch.stack(images).to(device)

with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

# print results
for path, result in zip(image_paths, results):
    print("\n====================")
    print("Image:", path)
    print(result)