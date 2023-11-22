import numpy as np
from ultralytics import YOLO
from pathlib import Path

BASE_PATH = Path.cwd()
DATA_PATH = rf'{BASE_PATH}/data/'
MODEL_PATH = rf'{BASE_PATH}/models/'
TESTING_PATH = rf'{BASE_PATH}/testing/'

def train(data_path: str, epochs: int = 10, image_size: int = 64) -> any:
    model = YOLO('yolov8n-cls.pt')  # loading a pretrained model (recommended for training)
    results = model.train(data=data_path, epochs=epochs, imgsz=image_size)
    print(results)
    return model

def save(model) -> None:
    model.export()

def predict(model_path: str, predict_image_path: list) -> None:
    model = YOLO(model_path)
    results = model(predict_image_path) 
    
    for result in results:
        names = result.names
        probs = result.probs
        print(names)
        print(probs.data.tolist())
        print(f'The given image is classified as: {names[probs.top1]}, whereas the image was {Path(result.path).stem}')
        print('\n')


def run(data_path: str, epochs: int, image_size: int):
    # For model training & saving
    model = train(data_path, epochs, image_size)
    save(model)
    
    # For model prediction - use the latest trained model weights for model_path below
    image_prediction_list = [f'{TESTING_PATH}building.jpeg',
                            f'{TESTING_PATH}forest.jpeg',
                            f'{TESTING_PATH}glacier.jpeg',
                            f'{TESTING_PATH}mountain.jpeg',
                            f'{TESTING_PATH}sea.jpeg',
                            f'{TESTING_PATH}street.jpeg']
    predict(model_path=f'{BASE_PATH}/runs/classify/train/weights/best.pt', predict_image_path=image_prediction_list)
 

if __name__ == '__main__':
    dataset_path = DATA_PATH
    epochs = 2
    image_size = 64

    run(dataset_path, epochs, image_size)
