import torch
import numpy as np

class OIObjModel():
    def __init__(self, logger, base_configs):
        self.logger = logger
        self.base_configs = base_configs

    def setup(self):
        model_name = self.base_configs['model_name']
        self.logger.debug(f'Will setup model: {model_name}')
        cpu_only = self.base_configs.get('cpu_only', False)
        cpu_device = torch.device('cpu')

        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.eval()
        self.class_labels = self.model.names

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available() and not cpu_only:
            self.model.to('cuda')
        else:
            self.logger.debug('On CPU')
            self.model.to(cpu_device)

        if self.base_configs.get('hot_start', False) is True:
            self.logger.debug('Running HOT start...')

            self.logger.debug(f'hot start result: {self._hot_start(225, 225)}')

    def _hot_start(self, width, height, bgr_color=(0, 0, 0)):
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)
        # Fill image with color
        image[:] = bgr_color
        return self.predict(image)

    def preprocess(self, input_image):
        # not needed?
        # rgb_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # 1280
        return input_image

    def predict(self, input_image):
        preprocess_img = self.preprocess(input_image)

        with torch.no_grad():
            output = self.model([preprocess_img])
            output_predictions = output.xyxy[0]

        np_predicts = output_predictions.cpu().numpy().astype("float32")

        return self.post_processing(np_predicts)

    def post_processing(self, np_predicts):
        predicted_object_list = []
        for row in np_predicts:
            xmin, ymin, xmax, ymax, conf, class_idx = row
            if conf > self.base_configs['detection_threshold']:
                label = self.class_labels[class_idx]
                obj = {
                    'class_id': int(class_idx),
                    'label': label,
                    'bounding_box': [int(i) for i in [xmin, ymin, xmax, ymax]],
                    'confidence': float(conf)
                }
                predicted_object_list.append(obj)
        return {
            'data': predicted_object_list
        }


