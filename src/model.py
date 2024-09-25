import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# CNN Model
def build_cnn(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Random Forest
def random_forest_model(X_train, y_train):
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
