import pandas as pd
from utils.GlobalVarGetter import GlobalVarGetter
import ast

class BaseLeoSync:
    def __init__(self, config, df:pd.DataFrame):
        df['sats'] = df['sats'].apply(lambda x: ast.literal_eval(x))
        df['real_idx'] = df['real_idx'].apply(lambda x: ast.literal_eval(x))
        df['train_time_indiv'] = df['train_time_indiv'].apply(lambda x: ast.literal_eval(x))
        df['start_idx'] = df['real_idx'].apply(lambda x: x[0])
        df['end_idx'] = df['real_idx'].apply(lambda x: x[-1])
        self.schedule_group = self.get_schedule(df)
        
    def next(self, current_t):
        return self.schedule_group['sats'].iloc[current_t],self.schedule_group['end_idx'].iloc[current_t]
    
    def get_schedule(self,df:pd.DataFrame):
        return df

    def get_status(self):
        #将 列表元素展开成series，再使用value_counts()统计每个每个元素出现的次数
        group_id_count = self.schedule_group['sats'].explode().value_counts()
        group_len_count = self.schedule_group['sats'].apply(len).value_counts()

class FIFOLeoSync(BaseLeoSync):
    def __init__(self, config, df:pd.DataFrame):
        BaseLeoSync.__init__(self, config, df)

    def get_schedule(self,df:pd.DataFrame):
        indexs = [0]
        end_time = df['real_idx'].iloc[0][-1]
        # inde
        for index, row in df.iterrows():
            if row['real_idx'][0] >= end_time:
                indexs.append(index)
                end_time = row['real_idx'][-1]

        return df.loc[indexs].reset_index()
    
class MostRoundsLeoSync(BaseLeoSync):
    def __init__(self, config, df:pd.DataFrame):
        BaseLeoSync.__init__(self, config, df)

    def get_schedule(self,df:pd.DataFrame):
        df.sort_values(by=['end_idx'], inplace=True)
        indexs = [0]
        end_time = df['real_idx'].iloc[0][-1]
        # inde
        for index, row in df.iterrows():
            if row['real_idx'][0] >= end_time:
                indexs.append(index)
                end_time = row['real_idx'][-1]

        return df.loc[indexs].reset_index()

    
class MostGroupsLeoSync(BaseLeoSync):
    # 加权区间调度问题
    def __init__(self, config, df:pd.DataFrame):
        BaseLeoSync.__init__(self, config, df)

    def get_schedule(self,df:pd.DataFrame):
        df.sort_values(by=['real_idx'][-1], inplace=True)
        indexs = [0]
        end_time = df['real_idx'].iloc[0][-1]
        # inde
        for index, row in df.iterrows():
            if row['real_idx'][0] >= end_time:
                indexs.append(index)
                end_time = row['real_idx'][-1]

        return df.loc[indexs].reset_index()
    
    # 查找不冲突的最后一个活动的索引
    def find_last_non_conflict(self, df):
        indices = [-1] * len(df)
        for j in range(len(df)):
            for i in range(j-1, -1, -1):
                if df.iloc[i]['Time'][1] <= df.iloc[j]['Time'][0]:
                    indices[j] = i
                    break
        return indices

    # 动态规划计算最大权值和
    def max_weight_schedule(self, df):
        indices = self.find_last_non_conflict( df)
        max_weights = [0] * len(df)
        schedule = [[] for _ in range(len(df))]
        
        for j in range(len(df)):
            incl_weight = df.iloc[j]['Weight'] + (max_weights[indices[j]] if indices[j] != -1 else 0)
            if incl_weight > max_weights[j-1]:
                max_weights[j] = incl_weight
                schedule[j] = schedule[indices[j]] + [df.iloc[j]['Activity']] if indices[j] != -1 else [df.iloc[j]['Activity']]
            else:
                max_weights[j] = max_weights[j-1]
                schedule[j] = schedule[j-1]
    
        return schedule[-1], max_weights[-1]






