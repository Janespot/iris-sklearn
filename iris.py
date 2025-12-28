# import iris dataset
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature names: ", feature_names)
print("Target names: ", target_names)
print("\nFirst 10 rows of iris data:\n", X[:10])
print("\nTotal features: ", len(X))

# Split iris dataset into training and testing datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.4, 
    random_state = 1
)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# train the model using KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)

y_pred = classifier_knn.predict(X_test)

# Compare actual response values, y_test, with predicted values, y_pred, to find the accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Provide sample data for the model to make prediction on
sample = [
    [5, 5, 3, 2], 
    [2, 4, 3, 5]
]

preds = classifier_knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions1: ", preds)
print("Predictions: ", pred_species)