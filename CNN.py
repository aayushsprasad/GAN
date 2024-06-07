import os
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow_hub as hub
import scipy.linalg  # Needed for sqrtm function

# Define the CNN discriminator model
def build_cnn_discriminator(input_shape):
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    return model

# Define the input shape for the images
input_shape = (128, 128, 3)


# Build the CNN discriminator model
cnn_discriminator = build_cnn_discriminator(input_shape)

optimizer = Adam(learning_rate=0.001)
loss_function = BinaryCrossentropy(from_logits=False)

cnn_discriminator.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
cnn_discriminator.summary()



train_dir = r'C:\Users\aayus\PycharmProjects\GAN\dataset\img_align_celeba'
val_dir = r'C:\Users\aayus\PycharmProjects\GAN\dataset\val'
test_dir = r'C:\Users\aayus\PycharmProjects\GAN\dataset\test'
image_size = (128, 128)
batch_size = 32
seed_value = 108

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=seed_value,
    validation_split=0.2,
    subset='training'
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=seed_value,
    validation_split=0.2,
    subset='validation'
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=seed_value
)

num_epochs = 10
history = cnn_discriminator.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

for epoch in range(num_epochs):
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            predictions = cnn_discriminator(images, training=True)
            loss = loss_function(labels, predictions)
        grads = tape.gradient(loss, cnn_discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, cnn_discriminator.trainable_variables))

        # Optional: Print loss every 'n' epochs or batches to monitor training
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

val_predictions = cnn_discriminator.predict(val_dataset)
val_labels = np.concatenate([y for x, y in val_dataset], axis=0)
val_predictions_binary = (val_predictions > 0.5).astype(int)

# Specify the expected classes if there are exactly two classes, 0 and 1
cm = confusion_matrix(val_labels, val_predictions_binary, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

def visualize_predictions(dataset, model, num_examples=5):
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(num_examples):
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Predicted: {predictions[i][0]:.2f}, Actual: {labels[i].numpy()}")
            plt.axis('off')
            plt.show()

visualize_predictions(val_dataset, cnn_discriminator)

def calculate_fid(real_images, fake_images):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_images.mean(axis=0), np.cov(real_images, rowvar=False)
    mu2, sigma2 = fake_images.mean(axis=0), np.cov(fake_images, rowvar=False)
    # Compute the squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # Compute the trace of the covariance matrices
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # Check for imaginary numbers and handle them
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def inception_score(images, batch_size=32, num_classes=1001):
    model_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'
    model = hub.KerasLayer(model_url)
    def get_inception_probs(batch):
        batch = tf.image.resize(batch, (299, 299))
        batch = tf.keras.applications.inception_v3.preprocess_input(batch)
        return model(batch)
    n_batches = int(np.ceil(images.shape[0] / batch_size))
    preds = []
    for i in range(n_batches):
        batch = images[i * batch_size:(i + 1) * batch_size]
        preds.append(get_inception_probs(batch))
    preds = tf.concat(preds, axis=0)
    kl_divergences = preds * (tf.math.log(preds) - tf.math.log(tf.reduce_mean(preds, axis=0)))
    kl_divergences = tf.reduce_sum(kl_divergences, axis=1)
    inception_score = tf.exp(tf.reduce_mean(kl_divergences))
    scores_std = tf.math.reduce_std(tf.exp(kl_divergences))
    return inception_score.numpy(), scores_std.numpy()

real_image_dir = r'C:\Users\aayus\PycharmProjects\GAN\dataset\Real'
fake_image_dir = r'C:\Users\aayus\PycharmProjects\GAN\dataset\Fake'
real_images = []
fake_images = []

def load_and_process_images(directory, target_size=(128, 128)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB
            img = img.resize(target_size)  # Resize images to the target size
            img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
            images.append(img_array)
    return np.array(images)


# Load and process real and fake images
real_images_array = load_and_process_images(real_image_dir)
fake_images_array = load_and_process_images(fake_image_dir)

print(f"Loaded {len(real_images_array)} real images.")
print(f"Loaded {len(fake_images_array)} fake images.")

def preprocess_images(images, target_size=(299, 299)):
    images_resized = tf.image.resize(images, target_size)
    # Preprocess based on Inception's expectations
    images_normalized = tf.keras.applications.inception_v3.preprocess_input(images_resized)
    return images_normalized

def get_activations(images, model_url='https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'):
    model = hub.KerasLayer(model_url, output_shape=[2048], trainable=False)
    preprocessed_images = preprocess_images(images)
    activations = model(preprocessed_images)
    return activations.numpy()

# Example Usage
# Assume 'real_images_array' and 'fake_images_array' are loaded and are in the correct format (i.e., numpy arrays of raw pixel values in [0, 255])

real_images_tensor = tf.convert_to_tensor(real_images_array, dtype=tf.float32)
fake_images_tensor = tf.convert_to_tensor(fake_images_array, dtype=tf.float32)

real_activations = get_activations(real_images_tensor)
fake_activations = get_activations(fake_images_tensor)
# Assuming you have real_activations and fake_activations prepared as needed
fid_score = calculate_fid(real_activations, fake_activations)
is_score, is_std = inception_score(fake_images_array)

print("FID:", fid_score)
print("Inception Score:", is_score, "Std Dev:", is_std)
