#Import torch library
import torch
import torch.nn as nn
import torchvision.models as Model
import torchvision.transforms as transforms
import os
from PIL import Image
import MobileVGG
from TwoStreamMobileNetV3L import TwoStreamMobileNetV3L

class LoadModel:
    def __init__(self, model_type, file_dir, class_type = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_dir = file_dir
        self.model_type = model_type
        self.class_type = class_type
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.image_size = (224,224)
        if self.class_type == 'sam-ddd':
            self.class_dict = {0 : "safe driving",
                1 : "drinking",
                2 : "talking on the phone - left",
                3 : "talking on the phone - right",
                4 : "texting - left",
                5 : "texting - right",
                6 : "Doing hair",
                7 : "Adjusting specs",
                8 : "Reaching behind",
                9 : "Sleeping"}
        if self.class_type == 'statefarm':
            self.class_dict = {0 : "safe driving",
                1 : "texting - right",
                2 : "talking on the phone - right",
                3 : "texting - left",
                4 : "talking on the phone - left",
                5 : "operating the radio",
                6 : "drinking",
                7 : "reaching behind",
                8 : "hair and makeup",
                9 : "talking to passenger"}
        
        if self.model_type == 'mobilenetv3l':
            self.model = Model.mobilenet_v3_large()
            num_features = self.model.classifier[0].in_features
            output_shape = len(self.class_dict)

            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=1280),
                nn.Hardswish(),
                nn.Dropout(0.5),
                nn.Linear(1280, output_shape),
            )

        if self.model_type == 'mobilevgg_front':
            self.model = MobileVGG.MobileNetVGG()

        if self.model_type == 'twostreammobilenet':
            self.model = TwoStreamMobileNetV3L()

        checkpoint_path = os.path.join(self.file_dir, 'my_checkpoint.pth.tar')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])


        # Ensure the model is in evaluation mode
        self.model.eval()
        # Move model to the appropriate device
        self.model.to(self.device)

    def model_inference(self, image1, image2=None, preprocess=None):
        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ])

        self.model.eval()

        def process_image(image):
            img = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
            img_t = preprocess(img)
            return torch.unsqueeze(img_t, 0).to(self.device)

        batch1_t = process_image(image1)

        if image2 is not None:
            batch2_t = process_image(image2)
            with torch.no_grad():
                y = self.model(batch1_t, batch2_t)
        else:
            with torch.no_grad():
                y = self.model(batch1_t)

        result = y.argmax(dim=1).item()
        pred_class = self.class_dict[result]

        return pred_class

