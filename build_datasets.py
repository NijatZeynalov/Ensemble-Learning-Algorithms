#import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

#warning library
import warnings
warnings.filterwarnings('ignore')

#build datasets
random_state = 42
noise_class = 0.0
n_samples = 1000
n_features = 2
n_classes = 3

noise_moon = 0.1
noise_class = 0.2
noise_circle = 0.0
X, y = make_classification(n_samples=n_samples, n_features = n_features,
                    n_classes = n_classes,n_repeated=0, n_redundant=0,
                    n_informative=n_features,
                    random_state=random_state,
                    n_clusters_per_class=1,
                    flip_y=noise_class)
data = pd.DataFrame(X)
data['target'] = y
plt.figure()
plt.title('Dataset - make classification')
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue = 'target', data = data)
plt.show()

data_classification = (X,y)

#Second dataset
moon = make_moons(n_samples = n_samples, noise=noise_moon, random_state = random_state)

data = pd.DataFrame(moon[0])
data['target'] = moon[1]
plt.figure()
plt.title('Dataset - make moons')
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue = 'target', data = data)
plt.show()

#Third dataset
circle = make_circles(n_samples = n_samples, factor = 0.5, noise=noise_circle, random_state = random_state)

data = pd.DataFrame(circle[0])
data['target'] = circle[1]
plt.figure()
plt.title('Dataset - make circles')
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue = 'target', data = data)
plt.show()