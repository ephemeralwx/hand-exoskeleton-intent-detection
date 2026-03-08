import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def load_data_keras(mat_filepath):
    print("Loading data...")
    mat_data = sio.loadmat(mat_filepath)
    data_cell = mat_data['data']
    labels_cell = mat_data['labels']

    X_list, y_list, groups_list = [], [], []

    for i in range(4):
        # mat file has nested cell arrays format
        sub_data = data_cell[0, i]
        sub_labels = labels_cell[0, i].flatten()
        for j in range(sub_data.shape[0]):
            X_list.append(sub_data[j, 0])
            y_list.append(sub_labels[j])
            groups_list.append(i)

    return np.array(X_list), np.array(y_list), np.array(groups_list)

def build_cnn_lstm(input_shape=(100, 8)):
    model = models.Sequential()
    
    # arbitrarily picked 5 then 3 for kernel sizes, seemed to work ok
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    
    # dropout 0.3 to fight overfit. might need tuning later
    model.add(layers.LSTM(100, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))
    
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, groups = load_data_keras('labeled data 4 subjects.mat')
    print(f"Dataset Shape: {X.shape}") 
    
    accuracies, f1_scores = [], []
    
    for test_subject in range(4):
        print(f"\n--- Training CNN-LSTM: Leaving out Subject {test_subject + 1} ---")
        
        train_idx = groups != test_subject
        test_idx = groups == test_subject
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # hacky way to balance classes to avoid model predicting only dominant class
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        total = len(y_train)
        weight_for_0 = (1 / num_neg) * (total / 2.0)
        weight_for_1 = (1 / num_pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        model = build_cnn_lstm(input_shape=(100, 8))
        
        # patience 10 is arbitrary but saves time
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("Training model (this may take a few minutes)...")
        model.fit(X_train, y_train, 
                  epochs=50, 
                  batch_size=64, 
                  validation_data=(X_test, y_test),
                  class_weight=class_weight,
                  callbacks=[early_stop], 
                  verbose=1)
        
        y_pred_probs = model.predict(X_test, verbose=0)
        # thresh 0.5 might need tuning later if recall too low
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Opening (0)", "Opening (1)"]))
        
    print("\n========================================")
    print(f"FINAL CNN-LSTM LOSO RESULTS")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f}")
    print("========================================\n")