import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models        import Sequential
from tensorflow.keras.layers        import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers    import Adam
from tensorflow.keras.callbacks     import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics                import classification_report, confusion_matrix
import seaborn as sns

# 1. Пути
base_dir       = os.path.dirname(os.path.abspath(__file__))
train_dir      = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir       = os.path.join(base_dir, "test")
external_dir   = os.path.join(base_dir, "external_images")
model_path     = os.path.join(base_dir, 'nike_adidas_cnn.keras')
plots_dir      = os.path.join(base_dir, "plots")

# создаём папку plots, если её нет
os.makedirs(plots_dir, exist_ok=True)

# 2. Параметры
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4

# 3. Генераторы
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,      target_size=IMG_SIZE,
                                               batch_size=BATCH_SIZE, class_mode='binary',
                                               shuffle=True)
validation_gen = val_datagen.flow_from_directory(validation_dir,
                                                 target_size=IMG_SIZE,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='binary',
                                                 shuffle=False)
test_gen = test_datagen.flow_from_directory(test_dir,
                                            target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary',
                                            shuffle=False)

# 4. Архитектура
model = Sequential([
    Input(shape=(*IMG_SIZE,3)),
    Conv2D(32,(3,3),activation='relu'), MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'), MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'), MaxPooling2D((2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=LR),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
]

# 6. Обучение
history = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 7. Сохранение модели
model.save(model_path)
print(f"Model saved to {model_path}")

# 8. Стандартные графики
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.show()

# 9. Графики переобучения (сохранение в PNG)
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_acc  = history.history['accuracy']
val_acc    = history.history['val_accuracy']
epochs     = range(1, len(train_loss)+1)

# Gap Loss
plt.figure(figsize=(8,4))
plt.plot(epochs, np.array(val_loss)-np.array(train_loss), marker='o')
plt.title('Gap between Validation and Training Loss')
plt.xlabel('Epoch'); plt.ylabel('Val Loss - Train Loss')
plt.axhline(0, color='gray', linestyle='--'); plt.grid(True)
loss_png = os.path.join(plots_dir, "gap_loss.png")
plt.savefig(loss_png, bbox_inches='tight')
plt.close()
print(f"Saved loss-gap plot → {loss_png}")

# Gap Accuracy
plt.figure(figsize=(8,4))
plt.plot(epochs, np.array(train_acc)-np.array(val_acc), marker='o')
plt.title('Gap between Training and Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Train Acc - Val Acc')
plt.axhline(0, color='gray', linestyle='--'); plt.grid(True)
acc_png = os.path.join(plots_dir, "gap_acc.png")
plt.savefig(acc_png, bbox_inches='tight')
plt.close()
print(f"Saved acc-gap plot  → {acc_png}")

# 10. Оценка на тесте
print("Evaluating on test set...")
probs = model.predict(test_gen)
preds = (probs>0.5).astype(int).reshape(-1)
trues = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("Classification Report:")
print(classification_report(trues, preds, target_names=labels))

# 11. Матрица ошибок
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.show()

# 12. Функция-предсказатель
def predict_image(path, model, img_size=IMG_SIZE, labels=labels):
    img = load_img(path, target_size=img_size)
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)
    prob = model.predict(arr)[0][0]
    if prob > 0.5:
        label, conf = labels[1], prob*100
    else:
        label, conf = labels[0], (1-prob)*100
    plt.imshow(img); plt.title(f"{label} ({conf:.2f}%)"); plt.axis('off'); plt.show()
    return label, conf

# 13. Внешние картинки
print("External images predictions:")
for fname in os.listdir(external_dir):
    if not fname.lower().endswith(('.jpg','.png','.jpeg')):
        continue
    predict_image(os.path.join(external_dir, fname), model)

model.summary()
