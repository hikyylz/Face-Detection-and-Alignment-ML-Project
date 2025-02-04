import numpy as np
import cv2
import os
import json
from deepface import DeepFace

class CompressionClass:

    def pretrained_mesaurements(self, image1_path, image2_path):
        # image path den uuid değerini çek. o uuid li cropped image i bul. 
        cropped_image1_path = image1_path.split("Aligned_faces/aligned_face_")[1].split(".")[0] # uuid fetched
        cropped_image2_path = image2_path.split("Aligned_faces/aligned_face_")[1].split(".")[0]

        cropped_image1_path = "Cropped_faces/cropped_face_"+cropped_image1_path+".jpg"
        cropped_image2_path = "Cropped_faces/cropped_face_"+cropped_image2_path+".jpg"
        
        #deepface param tanımı
        model_name="VGG-Face"
        distance_metric1="cosine"
        distance_metric2="euclidean"

        # distance hesaplamaları
        respond_1 = DeepFace.verify(img1_path= cropped_image1_path, img2_path= cropped_image2_path, model_name= model_name, distance_metric= distance_metric1)
        respond_2 = DeepFace.verify(img1_path= cropped_image1_path, img2_path= cropped_image2_path, model_name= model_name, distance_metric= distance_metric2)

        cosine_distance = respond_1["distance"]
        euclidean_distance = respond_2["distance"]

        return cosine_distance, euclidean_distance

    def calculate_weighted_mse(self, image1_path, image2_path):
        #aligned image path geldi
        # Görüntüleri yükleyin
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE) # gri tonlama
        image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE) 

        if image1.shape != image2.shape:
            raise ValueError("Görüntü boyutlari ayni olmali!")

        # Görüntü boyutlarını alın
        h, w = image1.shape

        # Ağırlık matrisi oluştur (merkezdeki piksellere daha fazla ağırlık verilir)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        weight_matrix = 1.0 / (1.0 + ((y - center_y) ** 2 + (x - center_x) ** 2))

        # Piksel farklarının karesi
        squared_diff = (image1 - image2) ** 2

        # Ağırlıklı MSE hesaplama
        weighted_mse = np.sum(weight_matrix * squared_diff) / np.sum(weight_matrix)
        return weighted_mse
    

if __name__ == "__main__":
    import argparse

    # Argümanlari tanimla
    parser = argparse.ArgumentParser(description="Yüz karşilaştirma scripti.")
    parser.add_argument("image1_path", type=str, help="Girdi görüntü yolu.") 
    parser.add_argument("image2_path", type=str, help="Girdi görüntü yolu.")
    #parser.add_argument("json1_path", type=str, help="Yüz bilgilerini içeren JSON dosyasi yolu.")
    #parser.add_argument("json2_path", type=str, help="Yüz bilgilerini içeren JSON dosyasi yolu.")
    args = parser.parse_args()

    # Giriş dosyasi kontrolü
    if not os.path.exists(args.image1_path) or not os.path.exists(args.image2_path):
        print(f"Image dosya bulunamadi")
        exit(1) 

    """
    # Giriş dosyasi kontrolü
    if not os.path.exists(args.json1_path) or not os.path.exists(args.json2_path):
        print(f"Json dosya bulunamadi")
        exit(1)
    """

    comprassionEntity = CompressionClass()
    try:
        # image path den uuid değerini çek. o uuid li cropped image i bul. 
        image1_uuid = args.image1_path.split("Aligned_faces/aligned_face_")[1].split(".")[0] # uuid fetched
        image2_uuid = args.image2_path.split("Aligned_faces/aligned_face_")[1].split(".")[0]

        # aligned face i input olarak vermeliyim ve sadece iki tane insanın image ını vermeliyim MSE hesaplamak için.
        weighted_mse = comprassionEntity.calculate_weighted_mse(args.image1_path, args.image2_path)
        cosine_dist, euclidien_dist = comprassionEntity.pretrained_mesaurements(args.image1_path, args.image2_path)
        #aligned face input olarak al, uuid ile crop face i bul, pretrained e parametre olarak ver.

        comprison_results = {
            "comprison_results":[
                {
                    "method":"mse",
                    "value":weighted_mse
                },
                {
                    "method":"pretrained_embeddings",
                    "cosine_similarity": cosine_dist, #0.68
                    "euclidian_distance": euclidien_dist #1.17 bu değerlerden ufaksa true
                },
                {
                    "method":"distilled_embeddings", # yapmadım bu modeli
                    "cosine_similarity": 0.1,
                    "euclidian_distance": 0.1
                }
            ]
        }

        # JSON dosyasını kaydetmek için klasörü kontrol et ve oluştur
        json_folder = "Json_embeddings_files"
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)  # Klasör yoksa oluştur

        # JSON dosyasının tam yolu
        json_file_name = f"compresion_results_{image1_uuid}_{image2_uuid}.json"
        json_file_path = os.path.join(json_folder, json_file_name)

        # JSON dosyasını kaydet
        with open(json_file_path, 'w') as file:
            json.dump(comprison_results, file, indent=4)
            
        print(f"Compresion başariyla tamamlandi.")

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

