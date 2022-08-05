import io
import base64
import datetime
import onnx
import onnxruntime
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# model
MODEL_PATH = "2022-08-05_19:32:12_model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# classes
class_names = ['negative', 'positive']


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    img = transform_image(image_bytes=image_bytes)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out = ort_outs[0]
    predicted_idx = np.argmax(img_out[0])
    return class_names[predicted_idx]


def get_result(image_file, is_api=False):
    start_time = datetime.datetime.now()
    image_bytes = image_file.file.read()
    class_name = get_prediction(image_bytes)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'
    encoded_string = base64.b64encode(image_bytes)
    bs64 = encoded_string.decode('utf-8')
    image_data = f'data:image/jpeg;base64,{bs64}'
    result = {
        "inference_time": execution_time,
        "predictions": {
            "class_name": class_name
        }
    }
    if not is_api:
        result["image_data"] = image_data
    return result
