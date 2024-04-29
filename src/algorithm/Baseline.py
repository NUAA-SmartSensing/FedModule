from tracemalloc import start
import pandas as pd
from utils.GlobalVarGetter import GlobalVarGetter
import ast

class BaseLeoSync:
    def __init__(self, config, df:pd.DataFrame,period=90*60, sample_time=5):
        self.schedule_group = self.get_schedule(df)
        
    def next(self, current_t):
        return self.schedule_group['sats'].iloc[current_t-1],self.schedule_group['end_idx'].iloc[current_t-1]
    
    def get_schedule(self,df:pd.DataFrame):
        return df

    def get_status(self):
        #将 列表元素展开成series，再使用value_counts()统计每个每个元素出现的次数
        group_id_count = self.schedule_group['sats'].explode().value_counts()
        group_len_count = self.schedule_group['sats'].apply(len).value_counts()

        return len(self.schedule_group), group_id_count, group_len_count

class FedAvgLeoSync(BaseLeoSync):
    def __init__(self, config, df:pd.DataFrame,period=90* 60, sample_time=5):
        # df为sat与gs通信仿真的记录条数
        self.schedule_group = self.get_schedule(df,period, sample_time)


    def get_schedule(self,df:pd.DataFrame,period, sample_time):
        stop_time = df['end_idx'].max()
        step = int(period / sample_time)
        result_records = []
        count = 0
        max_idx = 0
        begin_time = 0
        while begin_time < stop_time:
            #  begin_time:end_time 为一个周期,找出该周期内出现至少两次的sat，并将最末的sat的end_idx作为下一个周期的起始时间，最大程度节省时间开销
            one_round = []
            sats = []
            train_time_indiv = []
            end_time = min(begin_time + step, stop_time)
            start_idx = end_time
            end_idx = begin_time

            # 筛选通信时隙在每个周期 begin_time和end_time之间的记录
            search_df = df[((df['start_idx'] >= begin_time) & (df['end_idx'] <= end_time)) | ((df['start_idx'] <= begin_time) & (df['end_idx'] > begin_time)) | ((df['start_idx'] < end_time) & (df['end_idx'] >= end_time))]
            for sat_id, sat_record in search_df.groupby('sat_id'):
                if len(sat_record) >= 2: # 该周期内至少有两次通信记录
                    count += 1
                    orbit = sat_record['Orbit'].iloc[0]
                    first_row = sat_record.iloc[0]  # 第一行
                    last_row = sat_record.iloc[-1]  # 最后一行

                    # 每增加一个sat，更新该 Round的start_idx和end_idx
                    start_idx = min(start_idx, first_row['end_idx'])
                    end_idx = max(end_idx, last_row['start_idx'])
                    
                    sats.append(sat_id)

                    # 训练时间设为第一次通信结束时间到最后一次通信开始时间
                    train_time_indiv.append((last_row['start_idx'] - first_row['end_idx']) * sample_time)
                    max_idx = max(max_idx, last_row['start_idx']) # 取最末一次通信开始时间为下一个同步周期开始时间


            begin_time = max_idx + 1
            if end_idx > start_idx:
                cost_time = (end_idx - begin_time) * sample_time
                result_records.append([sats, orbit, start_idx, end_idx, cost_time, train_time_indiv])
            if(len(sats) == 0 and stop_time - begin_time < step / 2):
                break
        print("Scheduler Compelted...")
        print(f"T of each round: {period}s")
        print("Total Rounds: ", len(result_records))
        print("Num of Sats Per Round on Average: ", count / len(result_records))
        return pd.DataFrame(result_records, columns=['sats', 'orbit', 'start_idx', 'end_idx', 'cost_time', 'train_time_indiv'])
    
class MostRoundsLeoSync(BaseLeoSync):
    def __init__(self, config, df:pd.DataFrame):
        BaseLeoSync.__init__(self, config, df)

    def get_schedule(self,df:pd.DataFrame):
        df.sort_values(by=['end_idx'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        indexs = [0]
        end_time = df['end_idx'].iloc[0]
        # inde
        for index, row in df.iterrows():
            if row['start_idx'] >= end_time:
                indexs.append(index)
                end_time = row['end_idx']

        return df.loc[indexs].reset_index()

    
class MostGroupsLeoSync(BaseLeoSync):
    # 加权区间调度问题
    def __init__(self, config, df:pd.DataFrame):
        BaseLeoSync.__init__(self, config, df)

    def get_schedule(self,df:pd.DataFrame):
        df.sort_values(by=['end_idx'], inplace=True)
        indexs = [0]
        end_time = df['end_idx'].iloc[0]
        # inde
        for index, row in df.iterrows():
            if row['start_idx'] >= end_time:
                indexs.append(index)
                end_time = row['end_idx']

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

