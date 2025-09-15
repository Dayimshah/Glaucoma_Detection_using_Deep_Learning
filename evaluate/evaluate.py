from sklearn.metrics import classification_report
import numpy as np

# Total number of batches in the generator
total_batches = len(testGen)
half_batches = total_batches // 4  # Use only half the batches

# Collect true and predicted labels
y_true, y_pred = [], []

for i, (batch_x, batch_y) in enumerate(test_gen_enhanced):
    if i >= half_batches:
        break
    # Predict the classification head
    predictions = model.predict(batch_x)
    pred_labels = (predictions[0] > 0.5).astype(int)
    y_pred.extend(pred_labels)
    y_true.extend(batch_y['classification'])

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Generate the classification report
report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])

print("Classification Report:")
print(report)
