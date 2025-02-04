import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FaceClusterer:
    def __init__(self, aligned_folder, cropped_folder, embeddings_folder):
        self.aligned_folder = aligned_folder
        self.cropped_folder = cropped_folder
        self.embeddings_folder = embeddings_folder

    def cluster(self, embeddings_folder):

        identities=[]

        for file in os.listdir(embeddings_folder):
            if file.endswith(".json"):
                comp_result_path = os.path.join(self.embeddings_folder, file)
                parts = comp_result_path.split("_")
                image1_ID = parts[4]
                image2_ID = parts[5].split(".")[0]

                with open(comp_result_path, "r") as comp_result:
                    data = json.load(comp_result)
                    pretarined_measures = data["comprison_results"][1]
                    cosine_dist = pretarined_measures["cosine_similarity"] # threashold values 0.68
                    euclidian_dist = pretarined_measures["euclidian_distance"] # 1.17

                    # sadece pretrained model çıktılarına göre karar veriyorum benzerliğe çünkü MSE doğru çalışıyor ve mobilenet hazır değil.
                    if cosine_dist < 0.68 and euclidian_dist < 1.17:
                        # bu iki insan aynı %100
                        similarty=100
                    elif cosine_dist < 0.68 or euclidian_dist < 1.17:
                        # %50 aynı insan
                        similarty=50
                    else:
                        # %0
                        similarty=0

                    identity = {
                            "person1_Id": image1_ID,
                            "person2_Id": image2_ID,
                            "similary": f"%{similarty}"
                        }

                identities.append(identity)
                    
        # identities json dosyası olarak kaydet
        json_file_name = "identities.json"
        # JSON dosyasını kaydet
        with open(json_file_name, 'w') as file:
            json.dump(identities, file, indent=4)

        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yüz kümeleme scripti.")
    parser.add_argument("aligned_folder", type=str, help="Hizalanmış yüz görüntülerinin klasörü.")
    parser.add_argument("cropped_folder", type=str, help="Kırpılmış yüz görüntülerinin klasörü.")
    parser.add_argument("embeddings_folder", type=str, help="Embedding JSON dosyalarının bulunduğu klasör.")
    args = parser.parse_args()

    clusterer = FaceClusterer(args.aligned_folder, args.cropped_folder, args.embeddings_folder)
    clusterer.cluster(args.embeddings_folder)

