import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np

def hamming_distance(image1, image2):
    return np.sum(image1 != image2)


def jaccard_similarity(image1, image2):
    intersection = np.sum(np.logical_and(image1, image2))
    union = np.sum(np.logical_or(image1, image2))
    return intersection / float(union)


def cosine_sim(image1, image2):
    image1 = image1.reshape(1, -1)
    image2 = image2.reshape(1, -1)
    return cosine_similarity(image1, image2)[0][0]


def calculate_similarity(image1, image2, type='hamming'):
    if type == 'hamming':
        return hamming_distance(image1, image2)
    elif type == 'jaccard':
        return jaccard_similarity(image1, image2)
    elif type == 'cosine':
        return cosine_sim(image1, image2)
    else:
        raise ValueError("Invalid similarity type specified.")


from sklearn.cluster import KMeans


def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    return w


def select_representative_images(all_patterns, max_patterns, type='hamming'):  # Added type parameter here
    selected_patterns = []
    n = len(all_patterns)

    # Initialize a similarity matrix
    similarity_matrix = np.zeros((n, n))

    # Calculate pairwise similarities between all patterns
    for i in range(n):
        for j in range(i, n):
            sim = calculate_similarity(all_patterns[i], all_patterns[j], type)  # Using type here
            similarity_matrix[i, j] = sim  # Fixed the syntax here, original could lead to issues
            similarity_matrix[j, i] = sim  # Ensuring symmetry

    # Clustering and Centroid Selection
    if max_patterns < n and max_patterns > 0:
        kmeans = KMeans(n_clusters=max_patterns).fit(similarity_matrix)
        centroids = kmeans.cluster_centers_
        # Select patterns closest to centroids
        for center in centroids:
            closest_pattern_idx = np.argmin(np.linalg.norm(similarity_matrix - center, axis=1))
            selected_patterns.append(all_patterns[closest_pattern_idx])

    return selected_patterns


def retrieve_pattern(weights, data, steps):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res


def retrieve_most_similar(image_size,weights, noisy_image, all_patterns, similarity_type='hamming'):

    noisy_image = np.ravel(noisy_image)
    reconstructed_image = retrieve_pattern(weights, noisy_image,image_size)
    
    # Calculate similarities
    similarities = [calculate_similarity(reconstructed_image, np.ravel(pattern), similarity_type) for pattern in
                    all_patterns]  # Ensure pattern is 1D array
    # print(f"Debug: Number of calculated similarities: {len(similarities)}")  # Debug print
    
    # Find most similar pattern
    most_similar_idx = np.argmax(similarities) if similarity_type == 'cosine' else np.argmin(similarities)
    most_similar_pattern = all_patterns[most_similar_idx]

    return most_similar_pattern


def subset_data_and_states(data, states, samples, num_samples_per_state):
    
    subset_data = []
    subset_states = []
    
    for state, indices in samples.items():
        # Take only the first 'num_samples_per_state' samples for each state
        selected_indices = indices[:num_samples_per_state]
        
        # Subset the data and states
        subset_data.append(data[selected_indices])
        subset_states.append(states[selected_indices])
    
    # Convert lists to numpy arrays and concatenate along the first axis
    subset_data = np.concatenate(subset_data, axis=0)
    subset_states = np.concatenate(subset_states, axis=0)
    
    return subset_data, subset_states


def train_and_evaluate_model(dir_name,train_images, test_images, image_size,err_percentage, n_epochs):
    # Preprocessing
    unique_labels = list(set([label for _, label in train_images + test_images]))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    train_images_array = np.array([img for img, _ in train_images])
    train_labels_array = np.array([label_to_int[label] for _, label in train_images])

    test_images_array = np.array([img for img, _ in test_images])
    test_labels_array = np.array([label_to_int[label] for _, label in test_images])

    # Model
    model = tf.keras.Sequential([
    tf.keras.layers.Reshape((image_size, image_size, 1), input_shape=(image_size, image_size)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    # Training
    history = model.fit(
        train_images_array, train_labels_array, 
        epochs=n_epochs,
        validation_split=0.1  # 1% of the training data is used as the validation set
    )

    # Evaluation
    test_loss, test_acc = model.evaluate(test_images_array, test_labels_array)
    print(f'Test accuracy: {test_acc}')

    # Confusion Matrix with Normalization
    predicted_labels = model.predict(test_images_array)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    cm = confusion_matrix(test_labels_array, predicted_labels)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting the normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    #save_path = f'{dir_name}/cm_imgsiz{image_size}_epoch{n_epochs}_err{err_percentage}.svg'
    #plt.savefig(save_path, dpi=300, format='svg')
    plt.show()
    
    return test_acc, cm_normalized


def create_binary_image(time_series, image_size, global_min=0, global_max=1):
    normalized_ts = (time_series - global_min) / (global_max - global_min)
    binary_image = -np.ones((image_size, image_size))
    for i, value in enumerate(normalized_ts):
        y_coord = int(value * (image_size - 1))
        x_coord = int(i * (image_size - 1) / (len(time_series) - 1))
        binary_image[(image_size - 1) - y_coord, x_coord] = 1
    return binary_image

def plot_sample_images(dir_name,images_dict, num_samples=4):
    #num_samples is the number of samples we plot for each state
    fig, axes = plt.subplots(len(images_dict.keys()), num_samples, figsize=(5, 5))

    for i, (state, images) in enumerate(images_dict.items()):
        random_samples = random.sample(images, min(num_samples, len(images)))
        for j, img in enumerate(random_samples):
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"State {state}")
    #save_path = f'{dir_name}/plotSamples.svg'
    #plt.savefig(save_path, dpi=300, format='svg')
    plt.show()
    
def prepare_training_and_testing_data(dir_name,max_patterns, binary_images):
    num_states = len(binary_images.keys())
    patterns_per_state = (max_patterns // 3)

    train_images = []
    test_images = []
    
    for state, images in binary_images.items():
        
        random.shuffle(images)
        upper_bound_train = min(patterns_per_state, len(images))
        
        train_selected = images[:upper_bound_train]
        train_images += [(img, state) for img in train_selected]
        test_images += [(img, state) for img in test_selected]

    state_counts_train = Counter(label for _, label in train_images)
    print("Number of training templates for each state:")
    for state, count in state_counts_train.items():
        print(f"State {state}: {count} templates")

    state_counts_test = Counter(label for _, label in test_images)
    print("Number of testing templates for each state:")
    for state, count in state_counts_test.items():
        print(f"State {state}: {count} templates")

    print(f"Shape of train_images: {len(train_images)}")
    print(f"Shape of test_images: {len(test_images)}")

    train_dict = {}
    test_dict = {}
    for img, state in train_images:
        if state not in train_dict:
            train_dict[state] = []
        train_dict[state].append(img)

    for img, state in test_images:
        if state not in test_dict:
            test_dict[state] = []
        test_dict[state].append(img)

    print("Training Images:")
    plot_sample_images(dir_name,train_dict)
    print("Testing Images:")
    plot_sample_images(dir_name,test_dict)

    return train_images, test_images

def apply_artifact(image, flip_percentage):
    """
    Flips a percentage of bits in a given binary image to introduce artifacts.
    
    Parameters:
    image (numpy array): The binary image
    flip_percentage (float): The percentage of bits to flip, between 0 and 1
    
    Returns:
    numpy array: The modified image with artifacts
    """
    
    total_elements = image.size
    num_to_flip = int(total_elements * flip_percentage)
    flip_indices = np.random.choice(total_elements, num_to_flip, replace=False)
    flat_image = image.flatten()
    flat_image[flip_indices] = (-1)*flat_image[flip_indices]
    modified_image = flat_image.reshape(image.shape)

    return modified_image
    
def prepare_training_and_testing_dataCNN(dir, max_patterns, binary_images):
    num_states = len(binary_images.keys())
    patterns_per_state = int(0.1*len(binary_images['MA']))

    train_images = []
    test_images = []
    
    for state, images in binary_images.items():
        
        random.shuffle(images)
        upper_bound_train = min(patterns_per_state, len(images))

        train_selected = images[:upper_bound_train]
        train_images += [(img, state) for img in train_selected]
        test_selected = images[upper_bound_train:]
        test_images += [(img, state) for img in test_selected]

    state_counts_train = Counter(label for _, label in train_images)
    print("Number of training templates for each state:")
    for state, count in state_counts_train.items():
        print(f"State {state}: {count} templates")

    state_counts_test = Counter(label for _, label in test_images)
    print("Number of testing templates for each state:")
    for state, count in state_counts_test.items():
        print(f"State {state}: {count} templates")

    print(f"Shape of train_images: {len(train_images)}")
    print(f"Shape of test_images: {len(test_images)}")

    train_dict = {}
    test_dict = {}
    for img, state in train_images:
        if state not in train_dict:
            train_dict[state] = []
        train_dict[state].append(img)

    for img, state in test_images:
        if state not in test_dict:
            test_dict[state] = []
        test_dict[state].append(img)

    print("Training Images:")
    plot_sample_images(dir,train_dict)
    print("Testing Images:")
    plot_sample_images(dir,test_dict)

    return train_images, test_images


def plot_binary_images(binary_images):
    unique_states = list(binary_images.keys())
    fig, axes = plt.subplots(len(unique_states), 15, figsize=(20, 8))
    
    for i, state in enumerate(unique_states):
        imgs = binary_images[state]
    
        random_indices = np.random.choice(len(imgs), 15, replace=False)
        random_imgs = [imgs[idx] for idx in random_indices]
        
        for j, img in enumerate(random_imgs):
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            if j == 0:
                ax.set_ylabel(state)
            ax.axis('off')
        fig.text(0.1,0.75 - i * 0.16, state, ha='center', va='center', rotation='vertical', fontsize=8)

    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def pipeline(subj, num_samples_per_state, image_size, segment_length, err_percentage, n_epochs):
    dir_name = f"{subj}_{num_samples_per_state}_{image_size}_{n_epochs}_{err_percentage}"
    
    # Create the directory if it doesn't exist
    
    #if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)
    #    print(f"Directory {dir_name} created.")
    #else:
    #    print(f"Directory {dir_name} already exists.")

    # Load the subset data directly
    subset_data = np.load(f"data/subset_data_{subj}.npy")
    
    # Load the subset metadata directly
    with open(f"data/subset_metadata_{subj}.pkl", 'rb') as f:
        subset_metadata = pickle.load(f)
        subset_states = subset_metadata['GT']
    unique_states = np.unique(subset_states)
    samples = {state: np.where(subset_states == state)[0] for state in unique_states}
    
    global_min = np.min([np.min(subset_data[samples[state]]) for state in unique_states]) * 1.25
    global_max = np.max([np.max(subset_data[samples[state]]) for state in unique_states]) * 0.75
    
    binary_images_before_artifact = {state: [] for state in unique_states}

    for state, state_samples in samples.items():
        for sample_idx in state_samples:
            for i in range(0, len(subset_data[sample_idx]), segment_length):
                sub_series = subset_data[sample_idx][i:i + segment_length]
                if len(sub_series) == segment_length:
                    binary_img = create_binary_image(sub_series, image_size, global_min=global_min, global_max=global_max)
                    binary_images_before_artifact[state].append(binary_img)

    binary_images = {state: [] for state in unique_states}
    for state, images in binary_images_before_artifact.items():
        for image in images:
            image_with_artifact = apply_artifact(image, err_percentage)
            binary_images[state].append(image_with_artifact)

    n_neurons = image_size * image_size
    max_patterns = int(0.138 * n_neurons)
    
    train_images, test_images = prepare_training_and_testing_dataCNN(dir_name,max_patterns, binary_images)
    test_acc, cm_normalized = train_and_evaluate_model(dir_name,train_images, test_images, image_size=image_size,err_percentage=err_percentage, n_epochs=n_epochs)

    return test_acc, cm_normalized





def pipeline_hopfield_rec(subj, num_samples_per_state, image_size, segment_length, err_percentage, n_epochs):
    dir_name = f"hopfield{subj}_{num_samples_per_state}_{image_size}_{n_epochs}_{err_percentage}"

    # Create the directory if it doesn't exist
    #if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)
    #    print(f"Directory {dir_name} created.")
    #else:
    #    print(f"Directory {dir_name} already exists.")

    # Load the subset data directly
    subset_data = np.load(f"data/subset_data_{subj}.npy")
    
    # Load the subset metadata directly
    with open(f"data/subset_metadata_{subj}.pkl", 'rb') as f:
        subset_metadata = pickle.load(f)
        subset_states = subset_metadata['GT']
        
    unique_states = np.unique(subset_states)
    samples = {state: np.where(subset_states == state)[0] for state in unique_states}

    global_min = np.min([np.min(subset_data[samples[state]]) for state in unique_states]) * 1.25
    global_max = np.max([np.max(subset_data[samples[state]]) for state in unique_states]) * 0.75

    binary_images_before_artifact = {state: [] for state in unique_states}

    for state, state_samples in samples.items():
        for sample_idx in state_samples:
            for i in range(0, len(subset_data[sample_idx]), segment_length):
                sub_series = subset_data[sample_idx][i:i + segment_length]
                if len(sub_series) == segment_length:
                    binary_img = create_binary_image(sub_series, image_size, global_min=global_min,
                                                     global_max=global_max)
                    binary_images_before_artifact[state].append(binary_img)

    binary_images = {state: [] for state in unique_states}
    for state, images in binary_images_before_artifact.items():
        for image in images:
            image_with_artifact = apply_artifact(image, err_percentage)
            binary_images[state].append(image_with_artifact)

    n_neurons = image_size * image_size
    max_patterns = int(0.138 * n_neurons)

    # Select Representative Images
    print("Step 1: Selecting Representative Images...")
    selected_patterns = {}
    for state in ["awake", "slow_updown", "MA"]:
        print(f"Processing state: {state}")
        clean_patterns = binary_images[state]
        max_train_patterns = int(max_patterns / 3)
        print(f"Max training patterns for {state}: {max_train_patterns}")
        selected_clean_patterns = select_representative_images(clean_patterns, max_train_patterns)
        print(f"Selected {len(selected_clean_patterns)} patterns for {state}")
        selected_patterns[state] = selected_clean_patterns

    # Train Hopfield Network
    print("\nStep 2: Training Hopfield Network...")
    trained_weights = {}
    for state, clean_patterns in selected_patterns.items():
        print(f"Training for state: {state}")
        artifact_patterns = [binary_images[state][i] for i in range(len(clean_patterns))]
        combined_patterns = artifact_patterns
        print(f"Total training patterns for {state}: {len(combined_patterns)}")
        weights = train(n_neurons, combined_patterns)
        trained_weights[state] = weights

    # Reconstruct All Images
    print("\nStep 3: Reconstructing All Images...")
    reconstructed_images = {}

    # Create a list of all clean patterns from all states to be able to chose from all of them
    all_clean_patterns = []
    for state in ["awake", "slow_updown", "MA"]:
        all_clean_patterns.extend(binary_images_before_artifact[state])

    # Reconstruct images
    for state in ["awake", "slow_updown", "MA"]:
        print(f"Reconstructing images for state: {state}")
        reconstructed_state_images = []

        for idx, img in enumerate(binary_images[state]):
            # Allow most similar image to be selected from any state
            most_similar_img = retrieve_most_similar(image_size, trained_weights[state], img, all_clean_patterns)
            reconstructed_state_images.append(most_similar_img)

        reconstructed_images[state] = reconstructed_state_images
        print(f"Reconstructed {len(reconstructed_state_images)} images for {state}")

    # Plot Some Reconstructed Images
    print("\nStep 4: Plotting Reconstructed Images...")

    # Assuming that image_size * image_size = n_neurons
    image_size = int(n_neurons ** 0.5)

    for state, images in reconstructed_images.items():
        print(f"Plotting for state: {state}")
        plt.figure(figsize=(12, 12))
        random_indices = np.random.choice(len(images), size=min(4, len(images)), replace=False)

        for idx, img_idx in enumerate(random_indices):
            plt.subplot(1, 4, idx + 1)
            plt.title(f"State: {state}, Img: {img_idx}")
            plt.imshow(images[img_idx].reshape((image_size, image_size)), cmap='gray')

        plt.show()

    plot_binary_images(binary_images)
    train_images, test_images = prepare_training_and_testing_dataCNN(dir_name, max_patterns, binary_images)
    print("train and test shapes", len(train_images), len(test_images))
    test_acc, cm_normalized = train_and_evaluate_model(dir_name, train_images, test_images, image_size=image_size,
                                                       err_percentage=err_percentage, n_epochs=n_epochs)

    return test_acc, cm_normalized


def pipeline_noartifact(subj, num_samples_per_state, image_size, segment_length, err_percentage, n_epochs):
    dir_name = f"noartifact{subj}_{num_samples_per_state}_{image_size}_{n_epochs}_{err_percentage}"
    dir = f"noartifact{subj}"
    
    """    
    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir_name)
        print(f"Directory {dir} created.")
    else:
        print(f"Directory {dir} already exists.")
    """
    
    # Load the subset data directly
    subset_data = np.load(f"data/subset_data_{subj}.npy")
    
    # Load the subset metadata directly
    with open(f"data/subset_metadata_{subj}.pkl", 'rb') as f:
        subset_metadata = pickle.load(f)
        subset_states = subset_metadata['GT']
    
    unique_states = np.unique(subset_states)
    samples = {state: np.where(subset_states == state)[0] for state in unique_states}

    global_min = np.min([np.min(subset_data[samples[state]]) for state in unique_states]) * 1.25
    global_max = np.max([np.max(subset_data[samples[state]]) for state in unique_states]) * 0.75

    binary_images = {state: [] for state in unique_states}

    for state, state_samples in samples.items():
        for sample_idx in state_samples:
            for i in range(0, len(subset_data[sample_idx]), segment_length):
                sub_series = subset_data[sample_idx][i:i + segment_length]
                if len(sub_series) == segment_length:
                    binary_img = create_binary_image(sub_series, image_size, global_min=global_min,
                                                     global_max=global_max)
                    binary_images[state].append(binary_img)
        print(f"Generated {len(binary_images[state])} binary images for state {state}")

    n_neurons = image_size * image_size
    max_patterns = int(0.138 * n_neurons)

    train_images, test_images = prepare_training_and_testing_dataCNN(dir_name, max_patterns, binary_images)
    test_acc, cm_normalized = train_and_evaluate_model(dir_name, train_images, test_images, image_size=image_size,
                                                       err_percentage=err_percentage, n_epochs=n_epochs)

    return test_acc, cm_normalized
