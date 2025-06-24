import tensorflow as tf
from tensorflow.keras import models, layers
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile as tiff


def load_paths_and_labels(config):

    data_dir = config["data_dir"]
    class_names = config["class_names"]
    class_to_label = {name: i for i, name in enumerate(class_names)}
    
    tif_files = [str(p) for p in data_dir.glob('*/*.tif')]
    labels = [class_to_label[Path(p).parent.name] for p in tif_files]
    
    print(f"Found {len(tif_files)} image files.")
    
    return tif_files, labels

def load_image_from_path(path):

    path_str = path.numpy().decode('utf-8')
    image = tiff.imread(path_str).astype('float32')

    return image

def tf_preprocess_image_wrapper(path, label, config):

    image = tf.py_function(func=load_image_from_path, inp=[path], Tout=tf.float32)
    image.set_shape([None, None, 3])

    IMG_SIZE = tuple(config["image_size"][:2]) # Get (224, 224) from config
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.per_image_standardization(image)
    image.set_shape(config["image_size"])
    
    return image, label

def create_tf_dataset(paths, labels, config,is_training=True):
    
    AUTOTUNE = tf.data.AUTOTUNE # to work efficent and fast

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=config["random_seed"])

    dataset = dataset.map(lambda path, label: tf_preprocess_image_wrapper(path, label, config), 
                          num_parallel_calls=AUTOTUNE) #efficient and parallelized processing of every element indata pipeline,
                                                       #transforming 'raw data' (file paths) into 'training-ready data' (image tensors)
    
    dataset = dataset.batch(config["batch_size"])

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def create_simple_cnn(config):

    model = models.Sequential([
        layers.Input(shape=config["image_size"]),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(config["dense_units"], activation='relu'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(config["dropout_rate"]),
        layers.Dense(config["num_classes"], activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_and_save_history(history, config):
    """
    Plots the training and validation accuracy & loss from a Keras history object
    and saves the figure to a file specified in the config.

    Args:
        history: A Keras History object returned by the model.fit() method.
        config (dict): The central configuration dictionary.
    """
    # --- Data Retrieval ---
    # Extract the metrics from the history object
    # The .get() method is a safe way to access dictionary keys
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    # Check if there is anything to plot
    if not acc or not loss:
        print(" History object is empty or does not contain 'accuracy' and 'loss' keys. Cannot generate plot.")
        return

    epochs_range = range(len(acc))

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
    # Create a figure with two subplots, one for accuracy, one for loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Training History for: {config["model_name"]}', fontsize=18, weight='bold')

    # --- Accuracy Subplot ---
    ax1.plot(epochs_range, acc, label='Training Accuracy', color='dodgerblue', marker='o', linestyle='--')
    if val_acc: # Only plot validation accuracy if it exists
        ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='darkorange', marker='o')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim([min(plt.ylim()), 1]) # Set y-axis limit for accuracy

    # --- Loss Subplot ---
    ax2.plot(epochs_range, loss, label='Training Loss', color='dodgerblue', marker='o', linestyle='--')
    if val_loss: # Only plot validation loss if it exists
        ax2.plot(epochs_range, val_loss, label='Validation Loss', color='darkorange', marker='o')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    
    # --- Saving the Figure ---
    # Construct the full save path from the config dictionary
    save_path = config["save_dir_plots"] / f"{config['model_name']}_history_plot.png"
    
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    
    try:
        plt.savefig(save_path)
        print(f"✅ Plot successfully saved to: {save_path}")
    except Exception as e:
        print(f" Failed to save plot: {e}")
    
    # Display the plot
    plt.show()