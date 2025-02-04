Readme file

## How to install dependencies.

you must run these commands;

- conda install -c conda-forge retina-face
- conda install pytorch torchvision -c pytorch
- pip3 install opencv-python-headless
- pip install tf-keras
- pip install deepface
- pip install opencv-python opencv-contrib-python
- conda install conda-forge::matplotlib

## Note Bene:

I used conda enviorement for virtualization instead of venv. pip install command caused errors. I sometimes had to download same pagaces with conda and pip.

## How to run each script

detect-faces.py example command:
python detect-faces.py image1.jpg

align-faces.py example command:
python align-faces.py image1.jpg Json_files/image1.json

compare-faces.py example command:
python compare-faces.py Aligned_faces/aligned_face_7ed1391d.jpg Aligned_faces/aligned_face_8ea61008.jpg

cluster-face.py example command:
python cluster-faces.py Aligned_faces Cropped_faces Json_embeddings_files

## An overview of detection approach

Verilen örnek veri seetindeki fotoğraflarİ kullandİm. yüz yakalama işlemi için retinaface modelini kullanmayİ tercih ettim çünkü güçlü ve kullanİşlİ bir model olduğunu düşünüyorum. Dökümanda söylenen formatta çalştİrmak için argüman yapİsİnİ kurdum. yüz tanİma için bir sİnİf oluşturdum ve gerekli methotlarİ bu sİnİfİn içerisinde tanİmladİm. bu sayede temiz ve düzenli kod yazmayİ amaçladİm.

## An overview of alignment approach

İşlediğim verileri kİrpma ve hizalama işlemleri için modeller kullandİm. Yüz kİrpma işleminde padding ekledim çünkü karşİlaştİrma aşamasİnda yüzlerin bazİlarİ edge case yaratİp sorun çİkartİyordu, bunun önüne geçtim. Opencv kütüphanesinden hizzalamakla alakalİ methodlar kullandİm. transformation matrix i oluştururken verdiğiniz referans landmark lardan sadece gözler ve burun kullandİm çünkü hizalama işleminin böyle daha yüksek doğrulukta çalİştİğİnİ gözlemledim. Dökümanda tanİmlanan komutlara uydum ve yerine getirdim. Yeni dosyalarİ kaydederken UUId kullandİm çünkü verileri işlerken işime yaramasİnİ planladİm.

## An overview of comperation approach

Karşilaştirma işleminde sadece image path yeterli oldu. json dosyalarİnİ argüman olarak kullanmadİm çünkü kullanilmaya gerek kalmadi. MSE ve Pretrained model i kullanrak iki fotoğraf arasİnda hesaplamalar yapabildim. MSE değerinin doğruluğundan şüphe ediyorum. MSE hesabİ için aligned_face, pretrained model hesabİ için ise cropped_face dosyasİnİ kullandim. Hesaplamalar boyunca dosya kontrolllerini hep kontrol ettim ve uygun hata mesajlarİnİ vermeye çaliştim. Comperision_result sonuçlarini oluşturmak için yardİmcİ bir script den yardİm aldim ve tüm fotoğraflari karşilaştirdim, bazi işlenemez fotoğraflar hata firkatti.

## An overview of clustration approach

Kümeleme işlemini yapması için dökümanda istenilen argümanlari alan bir kod yazdım. Kümeleme işlemi için sinif oluşturdum ve içerisindeki methodlarla işlemleri çaliştirdim. comprasion-result dökümanı sadece iki tane insanın benzerliklerinin hesabını tutuyor. İlk olarak istenilen işlemi yapmak için comprasion-result un yeterli olamayacağını düşünmüştüm. istenilen png marix i nasıl oluşturacağımı bulamadım.
Girdi olarak aldığım compresion-result klasöründe dosyaları insanlara özgü atadığım uuid değerleriyle kaydettim. Compresion result lar arasında iterasyon yaparak her birinin kendi tanımladığım similarity score unu belirledim. Sonuçlarını idedtities.json dosyasına kaydettim.
