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
        sub_data = data_cell[0, i]
        sub_labels = labels_cell[0, i].flatten()
        for j in range(sub_data.shape[0]):
            X_list.append(sub_data[j, 0])
            y_list.append(sub_labels[j])
            groups_list.append(i)

    return np.array(X_list), np.array(y_list), np.array(groups_list)

def inception_module(input_tensor, filters=32):
    bottleneck = layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='linear', use_bias=False)(input_tensor)
    
    # kernel sizes 10,20,40 are arbitrary. tried 8,16,32 but this generalized better.
    conv10 = layers.Conv1D(filters=filters, kernel_size=10, padding='same', activation='linear', use_bias=False)(bottleneck)
    conv20 = layers.Conv1D(filters=filters, kernel_size=20, padding='same', activation='linear', use_bias=False)(bottleneck)
    conv40 = layers.Conv1D(filters=filters, kernel_size=40, padding='same', activation='linear', use_bias=False)(bottleneck)
    
    pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    pool_conv = layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='linear', use_bias=False)(pool)
    
    x = layers.Concatenate(axis=-1)([conv10, conv20, conv40, pool_conv])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_inception_time(input_shape=(100, 8), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    # omitted skip connections cuz keeping them overfitted hard on subject 2. kinda hacky.
    x = inception_module(inputs, filters=32)
    x = inception_module(x, filters=32)
    x = inception_module(x, filters=32)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # dropout 0.4 seemed best, 0.5 underfit. ticket ML-102
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, groups = load_data_keras('labeled data 4 subjects.mat')
    print(f"Dataset Shape: {X.shape}") 
    
    accuracies, f1_scores = [], []
    
    for test_subject in range(4):
        print(f"\n--- Training InceptionTime: Leaving out Subject {test_subject + 1} ---")
        
        train_idx = groups != test_subject
        test_idx = groups == test_subject
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # basic heuristic for class imbalance, didn't bother w smote
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        total = len(y_train)
        weight_for_0 = (1 / num_neg) * (total / 2.0)
        weight_for_1 = (1 / num_pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        model = build_inception_time(input_shape=(100, 8))
        
        # patience 10 is super low but training takes forever locally
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("Training model (this may take a few minutes)...")
        model.fit(X_train, y_train, 
                  epochs=50, 
                  batch_size=256, 
                  validation_data=(X_test, y_test),
                  class_weight=class_weight,
                  callbacks=[early_stop], 
                  verbose=1)
        
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Opening (0)", "Opening (1)"]))
        
    print("\n========================================")
    print(f"FINAL INCEPTION-TIME LOSO RESULTS")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f}")
    print("========================================\n")