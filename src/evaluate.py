from sklearn.metrics import classification_report

# Assuming 'model' and 'X_test', 'y_test' are available

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
