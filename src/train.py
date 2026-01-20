from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import joblib
import gzip
from sklearn.pipeline import Pipeline

from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "breast_cancer.csv"
data = pd.read_csv(data_path)

col = data.columns  # .columns gives columns names in data

y = data.diagnosis  # M or B

lst = ['Unnamed: 32', 'id', 'diagnosis']
x = data.drop(lst, axis=1)

print(x.isna().sum())

# correlation map
fig, ax = plt.subplots(figsize=(14, 14))

# Plot correlation heatmap
sns.heatmap(
    x.corr(),
    annot=True,
    linewidths=0.5,
    fmt=".1f",
    ax=ax,
    cmap="coolwarm"
)

# Title (optional but useful)
ax.set_title("Feature Correlation Heatmap")

# Save figure
plt.savefig(project_dir / "data" / "correlation_heatmap.png", bbox_inches="tight")

# Close figure (important)
plt.close(fig)

## As it can be seen in map heat figure radius_mean, perimeter_mean and area_mean are correlated with each other so we will use only area_mean.
## Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefor I only choose concavity_mean. Apart from these, radius_se, perimeter_se and area_se are correlated and I only use area_se. radius_worst, perimeter_worst and area_worst are correlated so I use area_worst. Compactness_worst, concavity_worst and concave points_worst so I use concavity_worst. Compactness_se, concavity_se and concave points_se so I use concavity_se. texture_mean and texture_worst are correlated and I use texture_mean. area_worst and area_mean are correlated, I use area_mean.

drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave_points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave_points_worst','compactness_se','concave_points_se','texture_worst','area_worst']
# selected_feature = ['texture_mean', 'area_mean', 'concavity_mean', 'area_se', 'concavity_worst']
x_1 = x.drop(columns=drop_list1, axis=1)
print(x_1.columns)

# correlation map
# Create figure
fig, ax = plt.subplots(figsize=(14, 14))

# Plot correlation heatmap
sns.heatmap(
    x_1.corr(),
    annot=True,
    linewidths=0.5,
    fmt=".1f",
    ax=ax,
    cmap="coolwarm"
)

# Title (optional but useful)
ax.set_title("Feature Correlation Heatmap")

# Save figure
plt.savefig(project_dir / "data" / "correlation_heatmap_after_dropping_the_correlated_features.png",
            bbox_inches="tight")

# Close figure (important)
plt.close(fig)

x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.2, random_state=42)

# find best scored 5 features
select_feature = SelectKBest(chi2, k=5)
clf_rf_f = RandomForestClassifier()

pipe = Pipeline([
    ("feature_selection", select_feature),
    ("classifier", clf_rf_f)
])

pipe.fit(x_train, y_train)

selected_features = pipe[:-1].get_feature_names_out()
ac_f = pipe.score(x_test, y_test) * 100
print("Accuracy after select feature is:", round(ac_f, 2))

# ########################################################################

# ########################################################################

x_2 = x_1[selected_features]
x_train, x_test, y_train, y_test = train_test_split(x_2, y, test_size=0.2, random_state=42)

clf_rf = RandomForestClassifier(random_state=43)
clf_rf = clf_rf.fit(x_train, y_train)

model_dir = project_dir / "model"
model_dir.mkdir(exist_ok=True)

with gzip.open(model_dir / "classification_breast_cancer.data.gz", "wb") as f:
    joblib.dump(clf_rf, f)

ac = accuracy_score(y_test, clf_rf.predict(x_test))
print('Accuracy is: ', round(ac * 100, 2))
cm = confusion_matrix(y_test, clf_rf.predict(x_test))
# Create figure
plt.figure(figsize=(6, 4))

# Plot heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# Labels
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save to file
plt.savefig(project_dir / "data" / "confusion_matrix.png", bbox_inches="tight")

# Close figure (important for scripts / servers)
plt.close()

