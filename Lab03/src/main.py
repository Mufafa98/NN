import numpy as np
import pickle
import math
import pandas as pd
from scipy.ndimage import affine_transform

INPUT_FOLDER = 'fii-nn-2025-homework-2'
INPUT_FILE = 'extended_mnist_'

input_size = 784
output_size = 10

default_expected = 1
array_type = np.float32


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

def softmax(z: np.ndarray) -> np.ndarray:
    # - np.max(...) because it can explode if values are to big
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cros_entropy(obtained: np.ndarray, expected: np.ndarray) -> float:
    obtained = np.clip(obtained, 1e-12, 1.0) 
    return -np.sum(expected * np.log(obtained)) / obtained.shape[0]

def forward_propagation(
    input_data: np.ndarray, 
    weights: np.ndarray, 
    bias: np.ndarray
) -> np.ndarray:
    return input_data @ weights + bias

def back_propagation(
        learning_rate: float, 
        expected: np.ndarray, 
        obtained: np.ndarray, 
        input_data: np.ndarray,
        weight: np.ndarray, 
        bias: np.ndarray,
    ):
    m = input_data.shape[0]
    dz = (obtained - expected) / m
    
    dw = input_data.T @ dz
    db = np.sum(dz, axis=0)
    
    weight -= learning_rate * dw
    bias -= learning_rate * db

    return weight, bias

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
    ):
    x_g_init =  math.sqrt(6 / (input_size + output_size)) # Normalized Xavier/Glorot Initialization
    weights = np.random.randn(input_size, output_size).astype(array_type) * x_g_init
    bias = np.zeros(output_size, dtype=array_type)
    
    best_accuracy = 0.0
    best_iteration = 0
    best_weights = None
    best_bias = None
    
    for epoch in range(1, max_iterations + 1):
        mini_batches = create_mini_batches(input_data, input_label, batch_size)

        batch_losses = []
        for (data, label) in mini_batches:
            z = forward_propagation(data, weights, bias)
            y_pred = softmax(z)
            
            batch_loss = cros_entropy(y_pred, label)
            batch_losses.append(batch_loss)
            
            weights, bias = back_propagation(
                lr,
                label,
                y_pred,
                data,
                weights,
                bias,
            )
        
        z_test = forward_propagation(test_data, weights, bias)
        y_test_probs = softmax(z_test)
        predictions = np.array(np.argmax(y_test_probs, axis=1))

        true_labels = np.argmax(test_label, axis=1)
        correct_predictions = np.sum(predictions == true_labels)
        total_predictions = test_data.shape[0]
        accuracy = (correct_predictions / total_predictions) * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_iteration = epoch
            best_weights = weights.copy()
            best_bias = bias.copy()

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        print(f'Epoch: {epoch}/{max_iterations} Accuracy: {accuracy:.2f}% Loss: {avg_loss:.4f} LR: {lr:.6f} Best {best_accuracy:.2f}% in epoch {best_iteration}')
    
    # Print best accuracy at the end
    print('\n===== Training Complete =====')
    print(f'Best accuracy: {best_accuracy:.2f}% (achieved at epoch {best_iteration})')
    
    model_data = {
        'weights': best_weights,
        'bias': best_bias,
        'accuracy': best_accuracy,
        'iteration': best_iteration,
    }
    with open(f'best_model_{best_accuracy:.2f}_accuracy.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

    return best_weights, best_bias

def normalize(input_data, test_data):
    train_mean = np.mean(input_data, axis=0)
    train_std = np.std(input_data, axis=0) + 1e-8

    input_data = (input_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return input_data, test_data

def generate_submission(test_data, w, b):
    z = forward_propagation(test_data, w, b)
    test_pred_probs = softmax(z)
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

    w,b = train(
        input_data=input_data, 
        input_label=input_label, 
        test_data=test_data, 
        test_label=test_label, 
        lr=0.01,
        batch_size=64,
        max_iterations=50
    )

    # generate_submission(test_data, w, b)