import os
import json
import cv2
import numpy as np
import uuid

class FaceAligner:

    def align_face(self, face_region, landmarks):
        # Affine dönüşüm için 3 ana landmark noktası seçiyoruz
        src_points = np.array([
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks["nose"]
        ], dtype="float32")

        # Hedef (ideal) landmark noktaları - yüzün dik durmasını sağlamak için
        dst_points = np.array([
            [192.98138, 239.94708],  # Sol göz 
            [318.90277, 240.1936],   # Sağ göz 
            [256.63416, 314.01935]   # Burun 
        ], dtype="float32")

        # Affine dönüşüm matrisini hesapla
        transform_matrix = cv2.getAffineTransform(src_points, dst_points)

        # Görüntüyü dönüştür ve 512x512 boyutuna hizala
        aligned_face = cv2.warpAffine(face_region, transform_matrix, (512, 512))

        return aligned_face, transform_matrix

    def process_faces(self, image_path, json_path):
        #outputs are stored in these folders
        Aligned_faces = "Aligned_faces"
        Cropped_faces = "Cropped_faces"

        # Girdi görüntüsünü ve yüz metadata'sini yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")

        with open(json_path, "r") as file:
            face_data = json.load(file)

        if not os.path.exists(Aligned_faces):
            os.makedirs(Aligned_faces)

        if not os.path.exists(Cropped_faces):
            os.makedirs(Cropped_faces)

        updated_faces = []
        for i, face in enumerate(face_data):
            bounding_box = face["bounding_box"]
            landmarks = face["landmarks"]
            face_uuid = str(uuid.uuid4())[:8]

            # Bounding box'u genişletmek için bir padding ekleyin
            padding = 20  # Genişletme miktarı
            x1, y1, x2, y2 = bounding_box

            # Yeni koordinatları hesapla (resim boyutlarının dışına çıkmamaya dikkat edin)
            x1 = max(0, x1 - padding)  # Sol kenarı sola kaydır
            y1 = max(0, y1 - padding)  # Üst kenarı yukarı kaydır
            x2 = min(image.shape[1], x2 + padding)  # Sağ kenarı sağa kaydır
            y2 = min(image.shape[0], y2 + padding)  # Alt kenarı aşağı kaydır

            # Genişletilmiş bounding box ile yüz bölgesini crop
            face_region = image[y1:y2, x1:x2]
            
            # Kirpilmiş yüzü kaydet
            cropped_output_path = os.path.join(Cropped_faces, f"cropped_face_{face_uuid}.jpg")
            cv2.imwrite(cropped_output_path, face_region) 

            # Landmarklari kirpilmiş yüz için normalize et
            adjusted_landmarks = {
                key: [landmark[0] - x1, landmark[1] - y1] for key, landmark in landmarks.items()
            }

            # Yüz hizalama işlemi
            aligned_face, transform_matrix = self.align_face(face_region, adjusted_landmarks)

            # Hizalanmiş yüzü ayri bir dosya olarak kaydet
            output_path = os.path.join(Aligned_faces, f"aligned_face_{face_uuid}.jpg")
            cv2.imwrite(output_path, aligned_face)

            # JSON dosyasini güncelle
            updated_faces.append({
                "bounding_box": bounding_box,
                "landmarks": landmarks,
                "transform_matrix": transform_matrix.tolist()  # Dönüşüm parametrelerini ekle
            })

        # Güncellenmiş JSON metadata dosyasini güncelle
        output_json_path = os.path.join(json_path)
        with open(output_json_path, "w") as file:
            json.dump(updated_faces, file, indent=4)

        print(f"Hizalama tamamlandi. Çiktilar {Aligned_faces} dizinine kaydedildi.")

if __name__ == "__main__":
    import argparse

    # Argümanlari tanimla
    parser = argparse.ArgumentParser(description="Yüz hizalama scripti.")
    parser.add_argument("image_path", type=str, help="Girdi görüntü yolu.")
    parser.add_argument("json_path", type=str, help="Yüz bilgilerini içeren JSON dosyasi yolu.")
    args = parser.parse_args()

    # Yüz hizalama işlemini başlat
    aligner = FaceAligner()
    aligner.process_faces(args.image_path, args.json_path)
