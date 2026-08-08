import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_heatmap(y_test, y_pred):
    
    # 1. Calculate correct and incorrect guesses
    cm = confusion_matrix(y_test, y_pred)
    
    # 2. Plot the colored heatmap directly
    plt.matshow(cm, cmap='Blues')
    
    # 3. Add the color scale on the right
    plt.colorbar()

    # 4. Label the grid ticks with our actual class names
    plt.xticks([0, 1], ['Benign (2)', 'Malignant (4)'])
    plt.yticks([0, 1], ['Benign (2)', 'Malignant (4)'])

    # 5. Add simple text labels
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('KNN Prediction Heatmap')

    # 6. Display the graph
    plt.show()