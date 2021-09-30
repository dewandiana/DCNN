import csv, numpy
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Opening the dataset csv files
with open('labels.csv','r') as dest_f:
    lb = list(csv.reader(dest_f, delimiter = ',' ))

with open('images.csv','r') as dest_f: 
    img = list(csv.reader(dest_f, delimiter = ',' ))

#splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(img, lb, test_size=0.2)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=50)

#Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train.values.ravel())
#clf.fit(X_train,y_train)

clf.fit(X_train,  numpy.ravel(y_train,order='C'))
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Confusion Matrix:')
cm=metrics.confusion_matrix(y_test, y_pred)
print(cm)

import seaborn
import matplotlib.pyplot as plt
 
 
def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    seaborn.set(font_scale=1)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'},fmt='d')
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

labels=["Multi-limb","Single-limb"]
plot_confusion_matrix(cm, labels, "confusion_matrix.png")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
