"""Generate reproducible figures and results for the Titanic research project."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parents[1]; rng=np.random.default_rng(42)
FIG=ROOT/'figures'; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
for name in ['eda','statistics','modeling','bias_variance']:(FIG/name).mkdir(parents=True,exist_ok=True)
sns.set_theme(style='whitegrid',context='talk'); plt.rcParams['savefig.dpi']=300
def save(fig,folder,name): fig.savefig(FIG/folder/(name+'.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))
def fit_logistic_gd(X,y,lr=.05,max_iter=10000,tol=1e-8):
    """Minimize BCE; gradient is X.T @ (sigmoid(X theta)-y)/n."""
    theta=np.zeros(X.shape[1]); losses=[]
    for step in range(max_iter):
        p=sigmoid(X@theta); losses.append(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))); grad=X.T@(p-y)/len(y); theta-=lr*grad
        if np.linalg.norm(grad)<tol: break
    return theta,np.array(losses),step+1
df=pd.read_csv(ROOT/'titanic.csv'); df['FamilySize']=df.SibSp+df.Parch+1; df['IsAlone']=(df.FamilySize==1).astype(int)
fig,ax=plt.subplots(figsize=(8,5)); sns.pointplot(data=df,x='Pclass',y='Survived',hue='Sex',errorbar=('ci',95),ax=ax); ax.set(title='Observed survival interaction: class x sex',ylabel='Survival proportion'); save(fig,'eda','sex_class_interaction')
fig,ax=plt.subplots(figsize=(8,5)); sns.histplot(data=df,x='Age',hue='Survived',kde=True,stat='density',common_norm=False,ax=ax); ax.set(title='Age distributions by outcome'); save(fig,'eda','age_distributions')
fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(df.select_dtypes('number').corr(method='spearman'),annot=True,cmap='vlag',center=0,ax=ax); ax.set(title='Spearman correlations'); save(fig,'eda','spearman_correlation')
tab=pd.crosstab(df.Sex,df.Survived); chi2,p,_,_=stats.chi2_contingency(tab); w=df.loc[df.Sex=='female','Survived'].to_numpy(); m=df.loc[df.Sex=='male','Survived'].to_numpy(); boots=[rng.choice(w,len(w),True).mean()-rng.choice(m,len(m),True).mean() for _ in range(5000)]
age1=df.loc[df.Survived==1,'Age'].dropna(); age0=df.loc[df.Survived==0,'Age'].dropna(); u,up=stats.mannwhitneyu(age1,age0)
pd.DataFrame([{'sex_chi2':chi2,'sex_p':p,'risk_difference':w.mean()-m.mean(),'risk_difference_ci_low':np.quantile(boots,.025),'risk_difference_ci_high':np.quantile(boots,.975),'age_mann_whitney_p':up,'age_rank_biserial':1-2*u/(len(age1)*len(age0))}]).to_json(OUT/'inference_results.json',orient='records',indent=2)
y=df.pop('Survived'); X=df.drop(columns=['PassengerId','Name','Ticket','Cabin']); cat=['Sex','Embarked']; num=[c for c in X if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); log=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]); log.fit(Xtr,ytr); prob=log.predict_proba(Xte)[:,1]
metrics=[]
metrics.append({'model':'logistic','roc_auc':roc_auc_score(yte,prob),'average_precision':average_precision_score(yte,prob),'brier':brier_score_loss(yte,prob)})
pd.DataFrame(metrics).to_csv(OUT/'model_metrics.csv',index=False)
fpr,tpr,_=roc_curve(yte,prob); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(fpr,tpr,label=f'AUC={roc_auc_score(yte,prob):.3f}'); ax.plot([0,1],[0,1],'k--'); ax.legend(); ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Held-out ROC curve'); save(fig,'modeling','heldout_roc')
pr,re,_=precision_recall_curve(yte,prob); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(re,pr); ax.set(xlabel='Recall',ylabel='Precision',title='Held-out precision-recall curve'); save(fig,'modeling','heldout_precision_recall')
obs,avg=calibration_curve(yte,prob,n_bins=8); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(avg,obs,'o-'); ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Mean predicted probability',ylabel='Observed frequency',title='Calibration curve'); save(fig,'modeling','calibration_curve')
M=df[['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone']].fillna(df[['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone']].median()); M=(M-M.mean())/M.std(); theta,losses,steps=fit_logistic_gd(np.c_[np.ones(len(M)),M],y.to_numpy()); pd.DataFrame({'iteration':np.arange(1,len(losses)+1),'loss':losses}).to_csv(OUT/'manual_logistic_convergence.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(losses); ax.set(xlabel='Iteration',ylabel='Binary cross-entropy',title=f'From-scratch logistic convergence ({steps} iterations)'); save(fig,'modeling','manual_logistic_convergence')
sizes,tr,va=learning_curve(log,X,y,cv=5,scoring='roc_auc',train_sizes=np.linspace(.2,1,6)); fig,ax=plt.subplots(figsize=(8,5)); ax.plot(sizes,tr.mean(1),'o-',label='Train'); ax.plot(sizes,va.mean(1),'o-',label='Validation'); ax.legend(); ax.set(xlabel='Training examples',ylabel='ROC-AUC',title='Learning curve'); save(fig,'bias_variance','learning_curve')
print(json.dumps({'rows':len(y),'manual_iterations':steps,'figures':len(list(FIG.rglob('*.png')))},indent=2))
