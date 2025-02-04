import os
import json
import torch
from retinaface import RetinaFace # yüz tanima için retinaface model ini kullanacağim.
import numpy as np  # Numpy'yi ekliyoruz, çünkü dönüşüm yapacağiz.

# Yüz algilama sinifi
class FaceDetector:
    def __init__(self, model_name='resnet50'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_faces(self, image_path):
        # RetinaFace ile yüz algilama
        detections = RetinaFace.detect_faces(image_path)
        
        # Yüzlerin JSON formatinda uygun hale getirilmesi
        faces = []
        for key, detection in detections.items():
            box = detection['facial_area']
            landmarks = detection['landmarks']
            
            # Numpy veri tiplerini float'a dönüştürme
            faces.append({
                'bounding_box': [int(coord) if isinstance(coord, np.int64) else coord for coord in box],
                'landmarks': {
                    'left_eye': [self.convert_to_float(coord) for coord in landmarks['left_eye']],
                    'right_eye': [self.convert_to_float(coord) for coord in landmarks['right_eye']],
                    'nose': [self.convert_to_float(coord) for coord in landmarks['nose']],
                    'mouth_left': [self.convert_to_float(coord) for coord in landmarks['mouth_left']],
                    'mouth_right': [self.convert_to_float(coord) for coord in landmarks['mouth_right']]
                }
            })
        
        return faces

    def convert_to_float(self, value):
        #Verilen değeri float'a dönüştürme (numpy float türlerini yönetir).
        if isinstance(value, np.float32):
            return float(value)
        return value

if __name__ == "__main__":
    import argparse

    # Argümanlari tanimla
    parser = argparse.ArgumentParser(description="Yüz algilama scripti.")
    parser.add_argument("image_path", type=str, help="Girdi görüntü yolu.")
    args = parser.parse_args()

    output_dir = "Json_files"

    # Giriş dosyasi kontrolü
    if not os.path.exists(args.image_path):
        print(f"Dosya bulunamadi: {args.image_path}")
        exit(1)

    # Outputs folder creation 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    # Yüz algilayici oluştur
    faceDetector = FaceDetector()
    try:
        # Yüz algilama işlemini gerçekleştir
        faces = faceDetector.detect_faces(args.image_path)

        # JSON dosyasini oluştur
        json_file_name = os.path.splitext(args.image_path)[0] + '.json'
        output_path = os.path.join(output_dir, json_file_name)
        with open(output_path, 'w') as file:
            json.dump(faces, file, indent=4)

        print(f"Başariyla tamamlandi. Çikti JSON dosyasi: {output_path}")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
