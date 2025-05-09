import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt


def train_model_and_show_results():

    # Define paths
    base_dir = "food_classifier/split"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    #params
    IMG_SIZE : int = 224
    BATCH_SIZE : int = 32
    EPOCHS : int = 10

    #Load data
    print(f"\n✅ Loading split dataset")
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"\n✅ Loading training dataset")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"\n✅ Loading validation dataset")

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"\n✅ Loading test dataset")

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"\n✅ Loading MobileNetv2 model")

    # Load mobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False #Freeze base 

    print(f"\n✅ Building the CNN layers")

    # Build full model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    print(f"\n✅ Compiling the model")

    #compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n✅ Training model")

    #train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    print(f"\n✅ Evaluating model on test dataset")

    # evaluate on test set
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


    model.save("food_classifier/ivan_model.h5")
    plt.figure(figsize=(14,5))

    #Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')  
    plt.legend()
    plt.grid(True)



    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# convert keras model to tflite
def convert_model_to_tflite() :
    model = tf.keras.models.load_model("food_classifier/ivan_model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    print(f"\n✅ Saving tflite model")
    try :
        with open("food_classifier/ivan_model.tflite", "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        print(f"Error saving tflite model: {e}")
    
def get_label_names() :
    TRAIN_DIR = "food_classifier/split/train"
    LABELS_FILE = "food_classifier/labels.txt"

    # Get sorted list of class names.
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

    # write them to labels.txt
    with open(LABELS_FILE, "w") as f:
        for name in class_names:
            f.write(name  + "\n")
    
    print(f"\n✅ Labels saved to {LABELS_FILE}")



if __name__ == "__main__":
    get_label_names()