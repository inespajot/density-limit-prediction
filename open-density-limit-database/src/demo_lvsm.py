import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc

import sys 
sys.stdout = open("../outputs/lsvm.txt", "w")


df = pd.read_csv('../DL_DataFrame.csv')
features = ['density','plasma_current','elongation','minor_radius','toroidal_B_field','triangularity']
print(f"Training on {len(features)} features")

X = df[features].values
y = df['density_limit_phase'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler() #scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(random_state=42,max_iter=10000)
clf.fit(X_train_scaled,y_train)

y_score = clf.decision_function(X_test_scaled)

fpr, tpr, thresholds = roc_curve(y_test,y_score) #roc curves
roc_auc = auc(fpr,tpr)
print(f"Area Under the Curve (AUC): {roc_auc:.3f}")

plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Linear SVM Classifier")
plt.legend(loc="lower right")
plt.savefig("../outputs/lsvm_roc", dpi=300)

#find hyperplane
weights = clf.coef_[0]
bias = clf.intercept_[0]
print("Decision Boundary Form (in scaled space):")
terms = [f"({w:+.3f} * {name})" for w, name in zip(weights, features)]
eq = " + ".join(terms) + f" {bias:+.3f} = 0"
print(eq)

#visualise feature weights
importance = pd.DataFrame({
    'Feature':features,
    "Weight": clf.coef_[0]
}).sort_values(by="Weight", key=abs,ascending=True)

plt.figure(figsize=(12,4))
plt.barh(importance["Feature"],importance["Weight"])
plt.axvline(0,color='black')
plt.xlabel("LSVM Model Coefficient Weight (Scaled)")
plt.title("Feature Importance for Density Limit Classification")
plt.savefig("../outputs/lvsm_feature_importance.png",dpi=300)

#visusalise two most dominant features
y_idx, x_idx = np.argsort(np.abs(weights)[-2:])
stable_idx = y_test==0
unstable_idx = y_test==1

plt.figure()
plt.scatter(X_test_scaled[stable_idx, x_idx], X_test_scaled[stable_idx, y_idx],
            color="blue", edgecolors="k", s=25, label="Stable (test)")
plt.scatter(X_test_scaled[unstable_idx,x_idx],X_test_scaled[unstable_idx,y_idx],
            color="red", edgecolors="k", s=25, label="DL (test)")
x_vals = np.linspace(X_test_scaled[:, x_idx].min(), X_test_scaled[:, x_idx].max(), 100)
y_vals = -(weights[x_idx] * x_vals + bias) / weights[y_idx]
plt.plot(x_vals, y_vals, "k--", label="Hyperplane Slice")
plt.xlabel(f"{features[x_idx]} (scaled)")
plt.ylabel(f"{features[y_idx]} (scaled)")
plt.title("Stable vs. Density Limit (Weightier features)")
plt.savefig("../outputs/lsvm_top2_boundary.png", dpi=300)