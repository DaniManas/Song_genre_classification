from sklearn.model_selection import train_test_split
from src.model import build_cnn, random_forest_model

# Assuming 'features' and 'labels' are loaded with data

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train CNN
cnn_model = build_cnn(input_shape=(X_train.shape[1], 1))
cnn_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Train Random Forest
rf_model = random_forest_model(X_train, y_train)

# Combine predictions
y_pred_cnn = cnn_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Hybrid prediction (ensemble method)
final_predictions = 0.5 * y_pred_cnn + 0.5 * y_pred_rf
