import os
import subprocess
import itertools

def run_compare_faces_for_all_pairs(folder):
    """Klasördeki tüm .jpg dosyaları için ikili kombinasyonlarla compare-faces.py'yi çalıştırır."""
    # Aligned_faces klasöründeki tüm .jpg dosyalarını listele
    image_files = [f for f in os.listdir(folder)]
    
    # İkili kombinasyonları oluştur (her resim çifti için)
    image_pairs = itertools.combinations(image_files, 2)

    for img1, img2 in image_pairs:
        image1_path = os.path.join(folder, img1)
        image2_path = os.path.join(folder, img2)
        print(image1_path)
        print(image2_path)
        command = ["python", "compare-faces.py", image1_path, image2_path]
        
        subprocess.run(command, check=True)

if __name__ == "__main__":
    run_compare_faces_for_all_pairs("Aligned_faces")
