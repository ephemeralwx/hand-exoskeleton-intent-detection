import numpy as np
import scipy.io as sio
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sktime.transformations.panel.rocket import MiniRocket
import time

def load_and_format_data(mat_filepath):
    print("Loading data...")
    mat_data = sio.loadmat(mat_filepath)
    data_cell = mat_data['data']
    labels_cell = mat_data['labels']

    X_list, y_list, groups_list = [], [], []

    # hardcoded to 4 because of legacy matlab format, will break if more subjects added
    for i in range(4):
        sub_data = data_cell[0, i]
        sub_labels = labels_cell[0, i].flatten()
        
        for j in range(sub_data.shape[0]):
            window = sub_data[j, 0]
            # sktime forces channels-first format which is annoying but required
            X_list.append(window.T)
            y_list.append(sub_labels[j])
            groups_list.append(i)

    return np.array(X_list), np.array(y_list), np.array(groups_list)

if __name__ == "__main__":
    X, y, groups = load_and_format_data('labeled data 4 subjects.mat')
    print(f"Total dataset shape: {X.shape}")
    
    accuracies, f1_scores = [], []
    
    # avoiding random split to prevent data leakage from correlated subject samples
    for test_subject in range(4):
        print(f"\n--- Testing on Subject {test_subject + 1} ---")
        
        train_idx = groups != test_subject
        test_idx = groups == test_subject
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        start_time = time.time()
        
        print(f"[{time.strftime('%H:%M:%S')}] Applying MiniRocket Transform (this may take a bit on the first fold due to compilation)...")
        step_start = time.time()
        # stick with minirocket instead of full rocket due to excessive runtimes on local
        minirocket = MiniRocket()
        minirocket.fit(X_train)
        
        X_train_transform = minirocket.transform(X_train)
        X_test_transform = minirocket.transform(X_test)
        print(f"[{time.strftime('%H:%M:%S')}] Transform complete in {time.time() - step_start:.2f} seconds. Features shape: {X_train_transform.shape}")
        
        print(f"[{time.strftime('%H:%M:%S')}] Training RidgeClassifierCV (finding optimal alpha). This is the heavy math step...")
        step_start = time.time()
        # ridge cv handles class imbalance reasonably well and shrinks less useful features
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
        classifier.fit(X_train_transform, y_train)
        print(f"[{time.strftime('%H:%M:%S')}] Classifier training complete in {time.time() - step_start:.2f} seconds.")
        
        print(f"[{time.strftime('%H:%M:%S')}] Evaluating model...")
        y_pred = classifier.predict(X_test_transform)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Training & Inference Time: {time.time() - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Opening (0)", "Opening (1)"]))

    print("\n========================================")
    print(f"Final ROCKET LOSO results")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Average F1-score: {np.mean(f1_scores):.4f}")
    print("========================================\n")