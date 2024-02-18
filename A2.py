import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import optuna
from sklearn.model_selection import GroupKFold

print("Reading Files")



def strip_header(the_file):
    print("Strip Header")
    with open(the_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header = line
            else:
                break #stop when there are no more #
    the_header = header[1:].strip().split('\t')
    df = pd.read_csv(the_file,comment='#',names=the_header,sep='\t')
    return df



def group_by_queryid(dataset):
    print("Grouping By Query")
    dataset_grouped=dataset.groupby('QueryID',dropna=False)
    num_gps=len(dataset_grouped)
    empty_df=pd.DataFrame(columns=dataset.columns)
    empty_df2=pd.DataFrame(columns=dataset.columns)
    for group in list(dataset_grouped.groups.keys()):
        to_concat=dataset_grouped.get_group(group).reset_index(drop=True)
        empty_df=pd.concat([empty_df,to_concat],axis=0)
    empty_df=empty_df.reset_index()
    empty_df.drop(['index'],axis=1,inplace=True)
    return empty_df,num_gps


def splitter(dataset,test_size,num_grps):
    print("Splitting data")
    dataset_grouped=dataset.groupby('QueryID',dropna=False)
    train_df=pd.DataFrame(columns=dataset.columns)
    test_df=pd.DataFrame(columns=dataset.columns)
    groups_list=shuffle(list(dataset_grouped.groups.keys()),random_state=0)
    test_gp_size=int(num_grps*test_size)
    for group in groups_list[:num_grps-test_gp_size]:
        to_concat=dataset_grouped.get_group(group).reset_index(drop=True)
        train_df=pd.concat([train_df,to_concat],axis=0)
    for group in groups_list[num_grps-test_gp_size:]:
        to_concat=dataset_grouped.get_group(group).reset_index(drop=True)
        test_df=pd.concat([test_df,to_concat],axis=0)
    train_df=train_df.reset_index()
    train_df.drop(['index'],axis=1,inplace=True)
    test_df=test_df.reset_index()
    test_df.drop(['index'],axis=1,inplace=True)
    return train_df,test_df



def ndcg_scorer(true_ds,preds):
    true_df=pd.DataFrame()
    true_df['QueryID']=true_ds['QueryID']
    true_df['Something']=np.zeros(len(true_ds['QueryID']),dtype='int')
    true_df['Docid']=true_ds['Docid']
    true_df['Label']=true_ds['Label']
    np.savetxt('true.txt',true_df.values,delimiter='\t',fmt='%s')
    pred_df=pd.DataFrame()
    pred_df['QueryID']=true_ds['QueryID']
    pred_df['Docid']=true_ds['Docid']
    pred_df['Preds']=preds
    np.savetxt('preds.txt',pred_df.values,delimiter='\t',fmt='%s')
    rf = read_sort_run(run_file)
    output_file = 'current.run'
    with open(output_file,'w') as f:
        f.write(rf)
    score=run_trec(this_os,qrel_file,output_file,verbose)
    return score

import subprocess
import platform
import os,sys

def get_ndcg_score(txt):
  for ln in txt.split('\n'):
    ln = ln.strip()
    fields = ln.split('\t')
    metric = fields[0].strip()
    if metric == 'ndcg':
      return float(fields[2])

def run_trec(this_os,qrel_file,run_file,verbose):
  if this_os == 'Linux':
    tbin = './trec_eval.linux'
  elif this_os == 'Windows':
    tbin = 'trec_eval.exe'
  elif this_os == 'Darwin':
    tbin = './trec_eval.osx'
  else:
    print('OS is not known')

  try:
    args = (tbin, "-m", "all_trec",
            qrel_file, run_file)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    txt = output.decode()
    if verbose == True:
      print (txt)
    else:
      return (get_ndcg_score(txt))
  except Exception as e:
    print('[ERROR]: subprocess failed')
    print('[ERROR]: {}'.format(e))
    
# NOTE: This is splitting on a space. Your output should use a tab
# So -- x,y,z = ln.split('\t')
def read_sort_run(run_file):
  qdic = {}
  lines = []
  with open(run_file,'r') as f:
    for ln in f:
      ln = ln.strip()
      #print (ln)
      x,y,z = ln.split('\t')
      if x in qdic:
        qdic[x].append((y,float(z)))
      else:
        qdic[x] = []
        qdic[x].append((y,float(z)))

  rank = 1
  for k,v in qdic.items():
    v.sort(key=lambda x:x[1], reverse=True)
    rank = 1
    for a,b in v:
        out = str(k) + ' Q0 ' + a + ' ' + str(rank) + ' ' + str(b) + ' e76767'
        lines.append(out)
        rank += 1
  return '\n'.join(lines)


verbose = False
this_os = platform.system()
qrel_file = 'true.txt'
run_file = 'preds.txt'
# print ('OS = ',this_os)
# print ('qrel file = ',qrel_file)
# print ('system run file = ',run_file)
# rf = read_sort_run(run_file)
output_file = 'current.run'
# with open(output_file,'w') as f:
#     f.write(rf)
# print(run_trec(this_os,qrel_file,output_file,verbose))

# redundant_columns=['QueryID', 'IDFBody', 'IDFAnchor', 'IDFTitle', 'IDFURL', 'IDFWholeDocument','Docid','Label','NumChildPages']

# trainX,valX=X.drop(redundant_columns,axis=1),y.drop(redundant_columns,axis=1)
# trainY,valY=X['Label'],y['Label']

# scaler=MinMaxScaler()
# trainX=scaler.fit_transform(trainX)
# valX=scaler.fit_transform(valX)

# initial_model=xgb.XGBRanker()
# groups = X.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
# initial_model.fit(trainX,trainY,group=groups,verbose=False)
# initial_preds=initial_model.predict(valX)
# initial_ndcg=ndcg_scorer(y,initial_preds)
# print("Initial NDCG Score: ",initial_ndcg)

#---------------------------------------------------OPTUNA HYP TUNING-----------------------------------------------------------
def parameter_sweep(trainX,trainY,valX,valY,y):
    #paraweepbegins
    print("Starting Parameter Tuning")
    final_params={}
    def objective_1(trial):
        params = {
                 'objective':trial.suggest_categorical('objective', ['rank:pairwise','rank:ndcg']),
                 'max_depth':trial.suggest_categorical('max_depth',[1,3,5,7,9]),
                 'min_child_weight':trial.suggest_categorical('min_child_weight',[1,3,5,7,9])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr
        
    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_1, n_trials=100)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 100
    # Best parameters: {'objective': 'rank:pairwise', 'max_depth': 3, 'min_child_weight': 1}}

    def objective_2(trial):
        params = {
                 'objective':'rank:pairwise',
                 'max_depth':trial.suggest_categorical('max_depth',[2,3,4]),
                 'min_child_weight':trial.suggest_categorical('min_child_weight',[0,0.2,0.4,0.6,0.8,1])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_2, n_trials=100)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 100
    # Best parameters: {'max_depth': 3, 'min_child_weight': 1}

    def objective_3(trial):
        params = {
                 'objective':'rank:pairwise',
                 'max_depth':3,
                 'min_child_weight':1,
                 'gamma':trial.suggest_categorical('gamma',[0,10,50,100,200,500])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_3, n_trials=20)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 20
    # Best parameters: {'gamma': 0}

    def objective_4(trial):
        params = {
                 'objective':'rank:pairwise',
                 'max_depth':3,
                 'min_child_weight':1,
                 'gamma':trial.suggest_categorical('gamma',[0,2,4,6,8])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_4, n_trials=10)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 10
    # Best parameters: {'gamma': 0}

    def objective_5(trial):
        params = {
                 'objective':'rank:pairwise',
                 'max_depth':3,
                 'min_child_weight':1,
                 'subsample':trial.suggest_categorical('subsample',[0,0.2,0.4,0.6,0.8,1]),
                 'colsample_bytree':trial.suggest_categorical('colsample_bytree',[0,0.2,0.4,0.6,0.8,1])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_5, n_trials=50)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 50
    # Best parameters: {'subsample': 0.8, 'colsample_bytree': 1}

    def objective_6(trial):
        params = {
                 'objective':'rank:pairwise',
                 'max_depth':3,
                 'min_child_weight':1,
                 'subsample':trial.suggest_categorical('subsample',[0.7,0.8,0.9]),
                 'colsample_bytree':trial.suggest_categorical('colsample_bytree',[0.9,1])
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_6, n_trials=20)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]

    # Trials Done: 20
    # Best parameters: {'subsample': 0.8, 'colsample_bytree': 1}

    def objective_7(trial):
        params = {
                 'eval_metric':trial.suggest_categorical('eval_metric',['map','ndcg']),
                 'objective':'rank:pairwise',
                 'max_depth':3,
                 'min_child_weight':1,
                 'subsample':0.8,
                 'colsample_bytree':1
                }
        model=xgb.XGBRanker(**params)
        model.fit(trainX,trainY,group=groups,verbose=False)
        preds=model.predict(valX)
        ndcg_scr=ndcg_scorer(y,preds)
        return ndcg_scr

    opt_study = optuna.create_study(direction='maximize')
    opt_study.optimize(objective_7, n_trials=5)
    print('Trials Done:', len(opt_study.trials))
    print('Best parameters:', opt_study.best_trial.params)
    for parameter in opt_study.best_trial.params:
        final_params[parameter]=opt_study.best_trial.params[parameter]
    # Trials Done: 5
    # Best parameters: {'eval_metric': 'ndcg'}
    return final_params
#parasweepends
#-----------------------------------------------------KFOLDVAL------------------------------------------------------------------

def kfold_val(final_params,train_final):
    print("\n\nStarting KFold Validation")
    redundant_cols2 = ['IDFBody', 'IDFAnchor', 'IDFTitle', 'IDFURL', 'IDFWholeDocument','NumChildPages']
    train_datakfold=train_final.drop(redundant_cols2,axis=1)
    X1=train_datakfold
    y1=train_datakfold['Label'].values
    gps=train_datakfold['QueryID'].copy().tolist()
    gfx=GroupKFold(n_splits=10)

    final_model=xgb.XGBRanker(**final_params)
    ndcg_scores=[]
    max_score=0
    i=1
    for train,test in gfx.split(X1,y1,gps):
        train_data = X1.iloc[train]
        test_data = X1.iloc[test]
        train_X=train_data.drop(['QueryID','Docid','Label'],axis=1)
        train_Y=train_data['Label']
        test_X=test_data.drop(['QueryID','Docid','Label'],axis=1)
        test_Y=test_data['Label']
        train_X=scaler.fit_transform(train_X)
        test_X=scaler.fit_transform(test_X)
        groups = train_data.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
        final_model.fit(train_X,train_Y,group=groups)
        preds=final_model.predict(test_X)
        score=ndcg_scorer(test_data,preds)
        ndcg_scores.append(score)
        mname="FinalModel.json"
        if score>max_score:
            final_model.save_model(mname)
        print("Fold",i,"Score: ",score)
        i=i+1
        
def final_predictor(test_dataset,model='Final_Tuned_Model.json'):
    print("creating predictions tsv ")
    new_final=xgb.XGBRanker()
    new_final.load_model(model)
    redundant_columns=['QueryID', 'IDFBody', 'IDFAnchor', 'IDFTitle', 'IDFURL', 'IDFWholeDocument','Docid','NumChildPages']
    final_test_data=test_dataset.drop(redundant_columns,axis=1)

    final_predictions=new_final.predict(final_test_data)

    final_data_frame=pd.DataFrame()

    final_data_frame['QueryID']=test_dataset['QueryID']
    final_data_frame['Docid']=test_dataset['Docid']
    final_data_frame['Score']=final_predictions

    final_data_frame.to_csv('s3882398.tsv',sep='\t',index=None,header=None)
    print("Final tsv generated, named:s3882398.tsv")
    
if __name__=="__main__":
    full_sweep=False
    if full_sweep==True:
        train_data=pd.read_csv('train.tsv',sep='\t')
        test_data=pd.read_csv('test.tsv',sep='\t')
        train_dataset = strip_header('train.tsv')
        test_dataset = strip_header('test.tsv')
        train_final,num_gps_trn = group_by_queryid(train_dataset)
        X,y=splitter(train_final,0.2,num_gps_trn)
        redundant_columns=['QueryID', 'IDFBody', 'IDFAnchor', 'IDFTitle', 'IDFURL', 'IDFWholeDocument','Docid','Label','NumChildPages']
        trainX,valX=X.drop(redundant_columns,axis=1),y.drop(redundant_columns,axis=1)
        trainY,valY=X['Label'],y['Label']
        scaler=MinMaxScaler()
        trainX=scaler.fit_transform(trainX)
        valX=scaler.fit_transform(valX)
        initial_model=xgb.XGBRanker()
        groups = X.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
        initial_model.fit(trainX,trainY,group=groups,verbose=False)
        initial_preds=initial_model.predict(valX)
        initial_ndcg=ndcg_scorer(y,initial_preds)
        print("Initial NDCG Score: ",initial_ndcg)
        params=parameter_sweep(trainX,trainY,valX,valY,y)
        kfold_val(params,train_final)
        final_predictor(test_dataset=test_dataset,model='FinalModel.json')
    else:
        test_data=pd.read_csv('test.tsv',sep='\t')
        test_dataset = strip_header('test.tsv')
        final_predictor(test_dataset=test_dataset)