import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import os
    import cv2
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # Load model and configs
    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Path to the folder containing test images
    test_images_folder = "/home/krishnadev/Pictures"

    # Get a list of all image files in the folder
    image_files = [os.path.join(test_images_folder, file) for file in os.listdir(test_images_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_files):
        # Load the image
        image = cv2.imread(image_path)

        # Perform prediction
        prediction_text = model.predict(image)

        # Extract ground truth label from the image filename
        label = os.path.splitext(os.path.basename(image_path))[0]

        # Compute Character Error Rate (CER)s
        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        # Display the resized image
        resized_image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Resized Image", resized_image)
        
        # Wait for a key press to move to the next image
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Calculate and print the average CER
    average_cer = np.average(accum_cer)
    print(f"Average CER: {average_cer}")

