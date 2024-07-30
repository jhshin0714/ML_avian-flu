# neutral pdb files directory = neu_1, pathogenic pdb files directory = highly_pathogen
from Bio.PDB import PDBParser
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 전체적인 PDB file에서 C-alpha 원자 좌표를 추출하는 function
def extract_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    return np.array(coords).flatten()

# pdb file들 가져오기.
neu_files = [os.path.join("neu_1", f) for f in os.listdir("neu_1") if f.endswith('.pdb')]
pathogen_files = [os.path.join("highly_pathogen", f) for f in os.listdir("highly_pathogen") if f.endswith('.pdb')]

#좌표 추출
neu_coords = [extract_coordinates(f) for f in neu_files]
pathogen_coords = [extract_coordinates(f) for f in pathogen_files]
all_coords = neu_coords + pathogen_coords
labels = ['neutral'] * len(neu_coords) + ['pathogenic'] * len(pathogen_coords)

# atom의 길이가 모두 다르기 떄문에, zero padding을 적용
max_length = max(len(coords) for coords in all_coords) # 모든 좌표 중 가장 긴 길이
padded_coords = []
for coords in all_coords:
    padding_length = max_length - len(coords)
    padded_coords.append(np.pad(coords, (0, padding_length), 'constant'))

# Data 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_coords = scaler.fit_transform(padded_coords)

#PCA 수행
pca = PCA(n_components=4)
principal_components = pca.fit_transform(padded_coords)
principalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])
principalDf['Class'] = labels
principalDf.to_csv('pca_results.csv', index=False) # PCA data 저장

# 모든 주성분의 조합을 통해 PCA plot을 그리기
import itertools
combinations = list(itertools.combinations(['PC1', 'PC2', 'PC3', 'PC4'], 2))
for comb in combinations:
    plt.figure(figsize=(10,6))
    for label, color in zip(['neutral', 'pathogenic'], ['blue','red']):
        mask = principalDf['Class']==label
        plt.scatter(principalDf[mask][comb[0]], principalDf[mask][comb[1]], label=label, c=color, alpha=0.5)
    plt.xlabel(comb[0])
    plt.ylabel(comb[1])
    plt.legend()
    plt.title(f'PCA of Protein structures using {comb[0]} and {comb[1]}')
    plt.show()

# abnormality detection 수행 및 ROC curve 그리기 (one - class SVM model)
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

data = pd.read_csv("pca_results.csv")
lb = LabelBinarizer()
data['binary_class'] = lb.fit_transform(data['Class'])
#레이블 반전이 필요할때
data['binary_class'] = [-1 if label == 'neutral' else 1 for label in data['Class']]

neutral_data = data[data['Class'] == 'neutral']
combinations = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC1', 'PC4'), 
                ('PC2', 'PC3'), ('PC2', 'PC4'), ('PC3', 'PC4')]
plt.figure(figsize=(10, 10))
for comb in combinations:
    clf= svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(neutral_data[list(comb)])
    scores = -clf.decision_function(data[list(comb)])
    fpr, tpr, _ = roc_curve(data['binary_class'], scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{comb[0]} & {comb[1]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for PC Combinations')
plt.legend(loc="lower right")
plt.show()

# 과적합 방지와 부트스트래핑을 통해 average ROC curve 그리기
## 부트스트래핑을 위한 함수
def bootstrap_sample(data, n=1):
    return [data.sample(n=len(data), replace=True) for _ in range(n)]

clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
plt.figure(figsize=(10, 8))
N_ITER = 100  # 부트스트래핑 반복 횟수

for comb in combinations:
    mean_fpr = np.linspace(0,1,100)
    tprs=[]
    for sample in bootstrap_sample(data, N_ITER):
        neutral_sample = sample[sample['Class']=='neutral']
        clf.fit(neutral_sample[list(comb)])
        fpr, tpr, _ = roc_curve(sample['binary_class'], scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{comb[0]} & {comb[1]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve from Bootstrapping for PC Combinations')
plt.legend(loc="lower right")
plt.show()

######## 최종적인 abnormality detection ROC curve
data = pd.read_csv('pca_results.csv')
lb = LabelBinarizer(neg_label=-1)
data['binary_class'] = lb.fit_transform(data['Class'])

# 색상 설정
color_map = {
    ('PC1', 'PC2'): '#0015FF',
    ('PC1', 'PC3'): '#FF00A1',
    ('PC1', 'PC4'): '#6BC800',
    ('PC2', 'PC3'): '#8400FF',
    ('PC2', 'PC4'): '#00BEB2',
    ('PC3', 'PC4'): '#FF7300'
}

combinations = list(color_map.keys())

# 부트스트래핑 함수
def bootstrap_sample(data, n=1):
    return [data.sample(n=len(data), replace=True) for _ in range(n)]

# ROC Curve 그리기
plt.figure(figsize=(10, 10))
N_ITER = 100
auc_scores = {}

for comb in combinations:
    mean_fpr = np.linspace(0,1,100)
    tprs=[]
    aucs = []
    for sample in bootstrap_sample(data, N_ITER):
        neutral_sample = sample[sample['Class']=='neutral']
        clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
        clf.fit(neutral_sample[list(comb)])
        scores = -clf.decision_function(data[list(comb)])
        fpr, tpr, _ = roc_curve(data['binary_class'], scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    auc_scores[comb] = mean_auc
    plt.plot(mean_fpr, mean_tpr, color=color_map[comb], label=f'{comb[0]} & {comb[1]} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color_map[comb], alpha=0.2)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve from Bootstrapping for PC Combinations')

# 레전드 순서를 변경하고 다시 플롯을 표시
order = np.argsort([-auc_scores[tuple(label.split(' (')[0].split(' & '))] if " & " in label else (-np.inf,) for label in labels])
ordered_handles = [handles[idx] for idx in order]
ordered_labels = [labels[idx] for idx in order]

# Random Guess가 이미 레전드에 있으면 추가하지 않음
if 'Random Guess' not in ordered_labels:
    ordered_handles.append(plt.Line2D([0], [0], color='navy', lw=2, linestyle='--'))
    ordered_labels.append('Random Guess')

plt.legend(ordered_handles, ordered_labels, loc="lower right")

plt.show()


#unsupervised learning 진행
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd

data = pd.read_csv('pca_results.csv')

combinations = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC1', 'PC4'), 
                ('PC2', 'PC3'), ('PC2', 'PC4'), ('PC3', 'PC4')]

param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 
              'weights': ['uniform', 'distance'], 
              'metric': ['euclidean', 'manhattan']}

N_ITER = 1000

results = []

for comb in combinations:
    print(f"Processing for combination: {comb}")
    
    X = data[list(comb)].values
    y = (data['Class'] == 'neutral').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    
    for _ in range(N_ITER):
        sample_X, sample_y = resample(X_train, y_train)
        best_knn.fit(sample_X, sample_y)
        y_pred = best_knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        
    results.append({
        'Combination': ' & '.join(comb),
        'Accuracy': f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'F1 Score': f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        'Recall': f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        'Precision': f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}"
    })

results_df = pd.DataFrame(results)
results_df.to_csv('knn_performance_results.csv', index=False)

