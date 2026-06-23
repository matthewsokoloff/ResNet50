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

resnet50 = resnet50.to(device).eval()

utils = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_convnets_processing_utils'
)

# imgs here
image_paths = [
    "images/hammer-test-1.jpg"
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
valid_paths = []

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        images.append(preprocess(img))
        valid_paths.append(path)
    except Exception as e:
        # crash prevention for failed img loads
        print(f"Failed to load {path}: {e}")

batch = torch.stack(images).to(device)

# use inference_mode bc it's faster than no_grad
with torch.inference_mode():
    # (*note: not apply softmax)
    output = resnet50(batch)

    # keep the logits stable; utils handle ranking
    results = utils.pick_n_best(predictions=output, n=5)

# print results
for path, result in zip(valid_paths, results):
    print("\n====================")
    print("Image:", path)
    print(result)