#Project LSI Dimitra_Kassari June 2023
#Part_A_Classification


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('bc.csv')
df.info()


row_zero_percentages = (df == 0).mean(axis=1)


threshold = 0.7


rows_to_drop = df.index[row_zero_percentages > threshold]


df_filtered = df.drop(rows_to_drop)


df_filtered.to_csv('bc_no0.csv', index=False)


dropped_rows = df.loc[rows_to_drop]
print("Dropped Rows:")
print(dropped_rows)


print("New Dataset:")
print(df_filtered)
df_filtered.info()



df = pd.read_csv('bc_no0.csv')
df.info()
df.isnull().sum()
df.describe()



df = pd.read_csv('bc_no0.csv')
class_proportions = df['gene_type'].value_counts(normalize=True)


imbalanced_threshold = 0.8  

if any(class_proportions > imbalanced_threshold):
    print("The dataset is imbalanced.")
else:
    print("The dataset is balanced.")



class_counts = df['gene_type'].value_counts()
class_proportions = class_counts / len(df)


print("Class Counts:")
print(class_counts)
print()
print("Class Proportions:")
print(class_proportions)



plt.figure(figsize=(10, 6))
plt.bar(class_proportions.index, class_proportions.values)
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.title('Class Distribution')
plt.xticks(rotation=90)
plt.show()


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class_counts = df['gene_type'].value_counts()


class_proportions = class_counts / len(df['gene_type'])
print("Class Proportions:")
print(class_proportions)


oversampler = RandomOverSampler()
X_over, y_over = oversampler.fit_resample(df.drop('gene_type', axis=1), df['gene_type'])
df_over = pd.concat([X_over, y_over], axis=1)


undersampler = RandomUnderSampler()
X_under, y_under = undersampler.fit_resample(df.drop('gene_type', axis=1), df['gene_type'])
df_under = pd.concat([X_under, y_under], axis=1)


class_counts_over = df_over['gene_type'].value_counts()
print("\nClass Counts after Oversampling:")
print(class_counts_over)


class_counts_under = df_under['gene_type'].value_counts()
print("\nClass Counts after Undersampling:")
print(class_counts_under)


# Split the dataset into features and target variable

X = df.drop(['gene_id','gene_name','gene_type'], axis=1)
y = df['gene_type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling if necessary
scaler = MinMaxScaler()  # or StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the shapes of the resulting datasets
print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('bc_no0.csv')

# Split the dataset into features and target variable
X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)
y = df['gene_type']

# Define the number of runs and folds
num_runs = 10
num_folds = 10

# Create empty lists to store metrics and confusion matrices
all_accuracies = []
all_recalls = []
all_specificities = []
all_precisions = []
all_f1_scores = []
all_confusion_matrices = []

# Perform the runs
for run in range(num_runs):
    # Create a new decision tree classifier
    dt = DecisionTreeClassifier()

    # Perform cross-validation
    kf = KFold(n_splits=num_folds, random_state=run, shuffle=True)
    fold_accuracies = []
    fold_recalls = []
    fold_specificities = []
    fold_precisions = []
    fold_f1_scores = []
    fold_confusion_matrices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the decision tree on the training data
        dt.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = dt.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        specificity = recall_score(y_test, y_pred,  average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)

        # Append the metrics and confusion matrix to the lists
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_specificities.append(specificity)
        fold_precisions.append(precision)
        fold_f1_scores.append(f1)
        fold_confusion_matrices.append(confusion)

    # Calculate average metrics for the run
    avg_accuracy = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_specificity = np.mean(fold_specificities)
    avg_precision = np.mean(fold_precisions)
    avg_f1_score = np.mean(fold_f1_scores)

    # Save the metrics to the general text file
    with open('dt_metrics.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        f.write(f"Accuracy: {avg_accuracy}\n")
        f.write(f"Recall: {avg_recall}\n")
        f.write(f"Specificity: {avg_specificity}\n")
        f.write(f"Precision: {avg_precision}\n")
        f.write(f"F1 Score: {avg_f1_score}\n\n")

    # Save the confusion matrices to the general text file
    with open('dt_confusion_matrix.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        for i, confusion in enumerate(fold_confusion_matrices):
            f.write(f"Fold {i+1}:\n")
            f.write(f"{confusion}\n\n")

    # Append the average metrics and confusion matrices to the lists
    all_accuracies.append(avg_accuracy)
    all_recalls.append(avg_recall)
    all_specificities.append(avg_specificity)
    all_precisions.append(avg_precision)
    all_f1_scores.append(avg_f1_score)
    all_confusion_matrices.append(fold_confusion_matrices)

# Print the confirmation message
print("Metrics saved to 'dt_metrics.txt'.")
print("Confusion matrices saved to 'dt_confusion_matrix.txt'.")


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

df = pd.read_csv('bc_no0.csv')

# Split the dataset into features and target variable

X = df.drop(['gene_id','gene_name','gene_type'], axis=1)
y = df['gene_type']


positive_class = 1

with open('knn_matrix_metrics.txt', 'w') as f, open('knn_metrics.txt', 'w') as f_metrics:
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        specificity = recall_score(y_test, y_pred,  average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')

        cm = confusion_matrix(y_test, y_pred)

        f.write(f"Run {i+1}:\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")

        cv_scores = cross_val_score(knn, X, y, cv=KFold(n_splits=10, random_state=i, shuffle=True))
        avg_cv_score = cv_scores.mean()
        f.write(f"10-Fold Cross Validation Scores (Run {i+1}):\n{cv_scores}\n")
        f.write(f"Average Cross Validation Score (Run {i+1}): {avg_cv_score}\n\n")

        f_metrics.write(f"Run {i+1}:\n")
        f_metrics.write(f"Accuracy: {accuracy}\n")
        f_metrics.write(f"Recall: {recall}\n")
        f_metrics.write(f"Specificity: {specificity}\n")
        f_metrics.write(f"Precision: {precision}\n")
        f_metrics.write(f"F1 Score: {f1}\n\n")

# Print the confirmation message
print("Metrics saved to 'knn_matrix_metrics.txt' and 'knn_metrics.txt'.")


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


df = pd.read_csv('bc_no0.csv')
X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)
y = df['gene_type']


num_runs = 10
num_folds = 10


all_accuracies = []
all_recalls = []
all_specificities = []
all_precisions = []
all_f1_scores = []
all_confusion_matrices = []


for run in range(num_runs):

    nb = GaussianNB()

 
    kf = KFold(n_splits=num_folds, random_state=run, shuffle=True)
    fold_accuracies = []
    fold_recalls = []
    fold_specificities = []
    fold_precisions = []
    fold_f1_scores = []
    fold_confusion_matrices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    
        nb.fit(X_train, y_train)

   
        y_pred = nb.predict(X_test)

  
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        specificity = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)

      
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_specificities.append(specificity)
        fold_precisions.append(precision)
        fold_f1_scores.append(f1)
        fold_confusion_matrices.append(confusion)


    avg_accuracy = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_specificity = np.mean(fold_specificities)
    avg_precision = np.mean(fold_precisions)
    avg_f1_score = np.mean(fold_f1_scores)

    
    with open('nb_metrics.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        f.write(f"Accuracy: {avg_accuracy}\n")
        f.write(f"Recall: {avg_recall}\n")
        f.write(f"Specificity: {avg_specificity}\n")
        f.write(f"Precision: {avg_precision}\n")
        f.write(f"F1 Score: {avg_f1_score}\n\n")

  
    with open('nb_confusion_matrix.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        for i, confusion in enumerate(fold_confusion_matrices):
            f.write(f"Fold {i+1}:\n")
            f.write(f"{confusion}\n\n")

   
    all_accuracies.append(avg_accuracy)
    all_recalls.append(avg_recall)
    all_specificities.append(avg_specificity)
    all_precisions.append(avg_precision)
    all_f1_scores.append(avg_f1_score)
    all_confusion_matrices.append(fold_confusion_matrices)


print("Metrics saved to 'nb_metrics.txt'.")
print("Confusion matrices saved to 'nb_confusion_matrix.txt'.")


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


df = pd.read_csv('bc_no0.csv')


X = df.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)
y = df['gene_type']


num_runs = 10
num_folds = 10


all_accuracies = []
all_recalls = []
all_specificities = []
all_precisions = []
all_f1_scores = []
all_confusion_matrices = []


for run in range(num_runs):
  
    svm = SVC()


    kf = KFold(n_splits=num_folds, random_state=run, shuffle=True)
    fold_accuracies = []
    fold_recalls = []
    fold_specificities = []
    fold_precisions = []
    fold_f1_scores = []
    fold_confusion_matrices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

     
        svm.fit(X_train, y_train)

   
        y_pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        specificity = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)

   
        fold_accuracies.append(accuracy)
        fold_recalls.append(recall)
        fold_specificities.append(specificity)
        fold_precisions.append(precision)
        fold_f1_scores.append(f1)
        fold_confusion_matrices.append(confusion)

    avg_accuracy = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_specificity = np.mean(fold_specificities)
    avg_precision = np.mean(fold_precisions)
    avg_f1_score = np.mean(fold_f1_scores)


    with open('svm_metrics.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        f.write(f"Accuracy: {avg_accuracy}\n")
        f.write(f"Recall: {avg_recall}\n")
        f.write(f"Specificity: {avg_specificity}\n")
        f.write(f"Precision: {avg_precision}\n")
        f.write(f"F1 Score: {avg_f1_score}\n\n")


    with open('svm_confusion_matrix.txt', 'a') as f:
        f.write(f"Run {run+1}:\n")
        for i, confusion in enumerate(fold_confusion_matrices):
            f.write(f"Fold {i+1}:\n")
            f.write(f"{confusion}\n\n")


    all_accuracies.append(avg_accuracy)
    all_recalls.append(avg_recall)
    all_specificities.append(avg_specificity)
    all_precisions.append(avg_precision)
    all_f1_scores.append(avg_f1_score)
    all_confusion_matrices.append(fold_confusion_matrices)


print("Metrics saved to 'svm_metrics.txt'.")
print("Confusion matrices saved to 'svm_confusion_matrix.txt'.")


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("f1_score.xlsx")


algorithms = ["KNN", "DT", "NB","SVM"]
columns = ["KNN", "DT", "NB","SVM"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("f1_score")


plt.title("F1_Score Comparison of Different Algorithms")
plt.savefig("f1_score_boxplot.png")

plt.show()




import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("accuracy.xlsx")


algorithms = [ "DT", "KNN","NB","SVM"]
columns = [ "DT", "KNN","NB","SVM"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("Accuracy")


plt.title("Accuracy Comparison of DT ,KNN , NB , SVM")
plt.savefig("accuracy_boxplot.png")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("specificity.xlsx")


algorithms = [ "DT", "KNN","NB","SVM"]
columns = [ "DT", "KNN","NB","SVM"]


accuracy = []


for col in columns:
    accuracy.append(df[col].values)


plt.boxplot(accuracy)
plt.xticks(range(1, len(algorithms) + 1), algorithms)
plt.ylabel("Specificity")


plt.title("Specificity Comparison of DT ,KNN , NB , SVM")
plt.savefig("specificity_boxplot.png")

plt.show()

