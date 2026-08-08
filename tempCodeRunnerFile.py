from Preprocess.Preprocessing import Preprocess_data
from Model.knn_classification import Knn_classification
from Visualization.Visuals import plot_heatmap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    data = Preprocess_data()
    X_train, X_test, y_train, y_test = data.load_preprocess_data()

    print("\n--- My Knn model ---")
    my_model = Knn_classification(n_neighbours=5)
    my_model.fit(X_train, y_train)
   
    
    print("Training Score:")
    y_pred_train = my_model.predict(X_train)
    print(my_model.accuracy(y_train, y_pred_train))
    
    print("Testing Score:")
    y_pred = my_model.predict(X_test)
    print(my_model.accuracy(y_test, y_pred))

    print("\n--- Inbuilt Knn model ---")
    inbuilt_model = KNeighborsClassifier(n_neighbors=5)
    inbuilt_model.fit(X_train, y_train)
    
    print("Training Score:")
    y_pred_train_inbuilt = inbuilt_model.predict(X_train)
    print(accuracy_score(y_train, y_pred_train_inbuilt))
    
    print("Testing Score:")
    y_pred_inbuilt = inbuilt_model.predict(X_test)
    print(accuracy_score(y_test, y_pred_inbuilt))
    
    print("\nGenerating Matplotlib Heatmap...")
    plot_heatmap(y_test, y_pred)

if __name__ == '__main__':
    main()