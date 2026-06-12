from pathlib import Path
import onnx
import os
import numpy as np

from advisor.face_parsing.utils.common import vis_parsing_maps
from advisor.face_parsing.onnx_inference import FaceParsingONNX
class FacialDetector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.num_classes = 19
        if model_name.lower() == 'resnet18':
            weight_path = Path(__file__).parent / 'face_parsing' / 'weights' / 'resnet18_weights.pth'
        elif model_name.lower() == 'resnet34':
            weight_path = Path(__file__).parent / 'face_parsing' / 'weights' / 'resnet34_weights.pth'
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'resnet18' and 'resnet34'.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_name, self.num_classes, weight_path=weight_path, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path):
        filename = os.path.basename(image_path)
        root_ext_pair = os.path.splitext(filename)
        save_raw_path = os.path.join(self.output_path, root_ext_pair[0] + '_raw.png')
        save_path = os.path.join(self.output_path, filename)

        try:
            image = Image.open(image_path).convert('RGB')
             
            original_size = image.size

            image_batch = prepare_image(image).to(self.device)

            output: torch.Tensor = self.model(image_batch)[0]
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

            mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
            restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

            # Save the raw mask
            mask_pil.save(save_raw_path)

            # Convert back to numpy array
            predicted_mask = np.array(restored_mask)

            # Visualize and save the results
            vis_parsing_maps(
                image,
                predicted_mask,
                save_image=True,
                save_path=save_path,
            )

        except Exception as e:
            print(f'Error processing image {image_path}: {e}')

if __name__ == "__main__":
    model_name = 'resnet18'
    image_path = 'test_face.jpg'  # Replace with your image path
    output_path = 'advisor/output'  # Replace with your desired output directory

    detector = FacialDetector(model_name)
    detector.output_path = output_path
    detector.process_image(image_path)