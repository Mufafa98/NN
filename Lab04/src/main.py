import numpy as np
import pickle
import math
import pandas as pd
from scipy.ndimage import affine_transform

INPUT_FOLDER = 'fii-nn-2025-homework-2'
INPUT_FILE = 'extended_mnist_'

input_size = 784
hidden_size = 100
output_size = 10

default_expected = 1
array_type = np.float32

# Data loading
def moments(image: np.ndarray):
    row_ind, col_ind = np.mgrid[:image.shape[0], :image.shape[1]]
    pixel_sum = np.sum(image)
    
    row_avg = np.sum(row_ind * image) / pixel_sum 
    col_avg = np.sum(col_ind * image) / pixel_sum 
    center_of_mass = np.array([row_avg, col_avg])
    
    # Variance and Covariance
    var_x = np.sum((row_ind - row_avg) ** 2 * image) / pixel_sum
    var_y = np.sum((col_ind - col_avg) ** 2 * image) / pixel_sum
    cov_xy = np.sum((row_ind - row_avg) * (col_ind - col_avg) * image) / pixel_sum

    covariance_matrix = np.array([
        [var_x, cov_xy], 
        [cov_xy, var_y]
        ])
    return center_of_mass, covariance_matrix

def deskew(image):
    center_of_mass, cov_matrix = moments(image)

    skew_factor = cov_matrix[0, 1] / cov_matrix[0, 0]
    affine = np.array([
        [1,             0], 
        [skew_factor,   1]
    ])

    image_center = np.array(image.shape) / 2.0
    offset = center_of_mass - np.dot(affine, image_center)

    img = affine_transform(image, affine, offset=offset)
    return (img - img.min()) / (img.max() - img.min())

def load_data(set: str) -> tuple[np.ndarray, np.ndarray]:
    with open(f'{INPUT_FOLDER}/{INPUT_FILE}{set}.pkl', 'rb') as f:
        obj = pickle.load(f)

    input_data = list[list[float]]()
    input_label = list[list[float]]()

    for entry in obj:
        image, label = entry
        np_image = np.array(image, dtype=array_type) 
        
        deskewed_image = deskew(np_image).reshape(-1) 
        input_data.append(deskewed_image)
        
        temp = list([0] * output_size)
        temp[label] = default_expected
        input_label.append(temp)

    input_data = np.array(input_data, dtype=array_type)
    input_label = np.array(input_label, dtype=array_type)

    return input_data, input_label

# Activation Functions
def softmax(z: np.ndarray) -> np.ndarray:
    # - np.max(...) because it can explode if values are to big
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)
# Derivative of relu
def d_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(array_type)

def cross_entropy(input, expected):
    # Clip input to avoid log(0)
    eps = 1e-8
    input_clipped = np.clip(input, eps, 1.0 - eps)
    loss = -np.sum(expected * np.log(input_clipped)) / input.shape[0]
    return loss

def forward_propagation(
    input_data: np.ndarray, 
    w1: np.ndarray, 
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Hidden layer
    z1 = input_data @ w1 + b1
    a1 = relu(z1)
    
    # Output layer
    z2 = a1 @ w2 + b2
    return z1, a1, z2

def back_propagation(
        learning_rate: float, 
        expected: np.ndarray, 
        obtained: np.ndarray, 
        input_data: np.ndarray,
        z1: np.ndarray,
        a1: np.ndarray,
        w1: np.ndarray, 
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        l2_strength: np.float32
    ):
    m = input_data.shape[0]
    # Output layer
    dz2 = (obtained - expected) / m
    dw2 = a1.T @ dz2 + l2_strength * w2
    db2 = np.sum(dz2, axis=0)
    # Hidden layer
    dz1 = (dz2 @ w2.T) * d_relu(z1)
    dw1 = input_data.T @ dz1 + l2_strength * w1
    db1 = np.sum(dz1, axis=0)
    
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1

    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    return w1, b1, w2, b2

def create_mini_batches(data: np.ndarray, label: np.ndarray, batch_size: int):
    mini_batches = []
    n_samples = data.shape[0]
    
    indices = np.random.permutation(n_samples)
    data_shuffled = data[indices]
    label_shuffled = label[indices]
    
    num_complete_batches = n_samples // batch_size
    
    for i in range(num_complete_batches):
        mini_batch_data = data_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batch_label = label_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((mini_batch_data, mini_batch_label))
    
    # Edge case when n_samples < batch_size
    if n_samples % batch_size != 0:
        mini_batch_data = data_shuffled[num_complete_batches * batch_size:]
        mini_batch_label = label_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((mini_batch_data, mini_batch_label))
    
    return mini_batches

def train(
        input_data,
        input_label,
        test_data,
        test_label,
        max_iterations: int = 500,
        lr: float = 0.1,
        batch_size: int = 64, 
        l2_strength: float = 0.0
    ):
    # Normalized Xavier/Glorot Initialization
    x_g_init1 =  math.sqrt(6 / (input_size + hidden_size)) 
    w1 = np.random.randn(input_size, hidden_size).astype(array_type) * x_g_init1
    b1 = np.zeros(hidden_size, dtype=array_type)
    
    x_g_init2 =  math.sqrt(6 / (hidden_size + output_size))  
    w2 = np.random.randn(hidden_size, output_size).astype(array_type) * x_g_init2
    b2 = np.zeros(output_size, dtype=array_type)
    
    best_accuracy = 0.0
    best_iteration = 0
    best_w1, best_b1, best_w2, best_b2 = None, None, None, None
    
    for epoch in range(1, max_iterations + 1):
        mini_batches = create_mini_batches(input_data, input_label, batch_size)

        batch_losses = []
        for (data, label) in mini_batches:
            z1, a1, z2 = forward_propagation(data, w1, b1, w2, b2)
            y_pred = softmax(z2)
            
            batch_loss = cross_entropy(y_pred, label)
            # L2 Regularization
            l2_penalty = l2_strength * (np.sum(w1 ** 2) + np.sum(w2 ** 2)) / 2
            batch_loss += l2_penalty
            # -----------------
            batch_losses.append(batch_loss)
            
            w1, b1, w2, b2 = back_propagation(
                lr, label, y_pred, data, z1, a1, w1, b1, w2, b2, l2_strength
            )
        
        # Evaluation
        z1_test, a1_test, z2_test = forward_propagation(test_data, w1, b1, w2, b2)
        y_test_probs = softmax(z2_test)
        predictions = np.argmax(y_test_probs, axis=1)

        true_labels = np.argmax(test_label, axis=1)
        accuracy = (np.sum(predictions == true_labels) / test_data.shape[0]) * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_iteration = epoch
            best_w1, best_b1 = w1.copy(), b1.copy()
            best_w2, best_b2 = w2.copy(), b2.copy()

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        print(f'Epoch: {epoch}/{max_iterations} Accuracy: {accuracy:.2f}% Loss: {avg_loss:.4f} LR: {lr:.6f} Best {best_accuracy:.2f}% in epoch {best_iteration}')
    
    print('\n===== Training Complete =====')
    print(f'Best accuracy: {best_accuracy:.2f}% (achieved at epoch {best_iteration})')
    
    model_data = {
        'w1': best_w1, 'b1': best_b1,
        'w2': best_w2, 'b2': best_b2,
        'accuracy': best_accuracy,
        'iteration': best_iteration,
    }
    with open(f'best_model_{best_accuracy:.2f}_accuracy.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

    return best_w1, best_b1, best_w2, best_b2

def normalize(input_data, test_data):
    train_mean = np.mean(input_data, axis=0)
    train_std = np.std(input_data, axis=0) + 1e-8

    input_data = (input_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return input_data, test_data

def generate_submission(test_data, w1, b1, w2, b2):
    _, _, z2 = forward_propagation(test_data, w1, b1, w2, b2)
    test_pred_probs = softmax(z2)
    predictions = np.argmax(test_pred_probs, axis=1)

    submission = pd.DataFrame({
        'ID': np.arange(len(predictions)),
        'target': predictions
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    np.random.seed(314)
    input_data, input_label = load_data('train')
    test_data, test_label = load_data('test')

    input_data, test_data = normalize(input_data, test_data)

    w1, b1, w2, b2 = train(
        input_data=input_data, 
        input_label=input_label, 
        test_data=test_data, 
        test_label=test_label, 
        lr=0.01,
        batch_size=64,
        max_iterations=100,
        l2_strength=0.0005
    )

    generate_submission(test_data, w1, b1, w2, b2)