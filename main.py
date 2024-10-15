import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#dataseti çek
mnist = tf.keras.datasets.mnist

#verileri eğitim ve test kümelerine ayır
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#verileri yeniden şekillendir ve normalize et
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32')/255

#CNN modelini oluştur
model = models.Sequential()

#1. Konvülasyon katmanı (32 filtre , 3x3 çekirdek, ReLu aktivasyonu)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

#MaxPooling katmanı(2x2 pencere boyutu)
model.add(layers.MaxPooling2D((2, 2)))

#2. konvülasyon katmanı (64 filtre, 3x3 çekirdek , ReLu aktivasyonu)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#veriyi düzleştir (flatten), yani konvolüsyon katmanından tam bağlı katmana geçiş
model.add(layers.Flatten())

#tam bağlı(Dense) katman
model.add(layers.Dense(64,activation='relu'))

#Çıkış katmanı (10 sınıf (epoch) için Softmax ile normalizasyon)
model.add(layers.Dense(10, activation='softmax'))

#modeli derle(compile)
model.compile(optimizer='adam', #backprpagation için adam optimizer
              loss = 'sparse_categorical_crossentropy', #kayıp fonksiyonu
              metrics=['accuracy'] #doğruluk metriği
              )

#modeli eğit
history = model.fit(train_images, train_labels, epochs=15,
                    validation_data=(test_images, test_labels))

#test seti üzerindeki doğruluğu kontrol et
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=2)
print(f'Test accuracy: {test_acc}')

#eğitim doğruluğu ve doğrulama doğruluğunu grafikle göster
plt.plot(history.history['accuracy'], label='Eğitim doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.show()

