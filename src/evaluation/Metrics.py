import pandas as pd
import numpy as np

class Metrics:

    def __init__(self, csv_path):
        self.extract_metdata_from_filename(csv_path)
        self.calulate_all_metrics(csv_path)
    
    def extract_metdata_from_filename(self, filename):
        name = filename.stem
        parts = name.split('_')
        metadata = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=')
                metadata[key] = value
            elif 'x' in part:
                metadata['map_size'] = part
        self.metadata = metadata

    def calulate_all_metrics(self, csv_path):
        self.df = pd.read_csv(csv_path)
        episodes = self.df.groupby('iteration')

        metrics_list = []
        counter = 0
        for iter_num, iter in episodes:
            
            # if one iteration failed because of errors etc, note iteration=episode
            if counter != iter_num:
                counter =self.append_missing_iterations(metrics_list, counter, iter_num)

            metrics = {
                'iteration': iter_num,
                
                # SUCCESS METRICS
                'success': self.is_successful(iter),
                'reached_goal': self.reached_goal(iter),
                
                # EFFICIENCY METRICS
                'num_steps': self.num_steps(iter),
                'steps_efficiency': self.steps_efficiency(iter),
                
                # REWARD METRICS
                'total_reward': self.total_reward(iter),
                'final_reward': self.final_reward(iter),
                'reward_per_step': self.reward_per_step(iter),
                
                # TERMINATION METRICS
                'termination_type': self.termination_type(iter),
            }
            metrics_list.append(metrics)
            counter += 1

        df_metrics = pd.DataFrame(metrics_list)

        # cumulative/rolling metrics with all (including failed) iterations
        df_metrics['cumulative_success_rate'] = df_metrics['success'].expanding().mean()
        df_metrics['rolling_success_rate_3'] = df_metrics['success'].rolling(3, min_periods=1).mean()
        df_metrics['rolling_success_rate_5'] = df_metrics['success'].rolling(5, min_periods=1).mean()
        df_metrics['rolling_success_rate_10'] = df_metrics['success'].rolling(10, min_periods=1).mean()
        df_metrics['rolling_avg_steps_3'] = df_metrics['num_steps'].rolling(3, min_periods=1).mean()
        df_metrics['rolling_avg_steps_5'] = df_metrics['num_steps'].rolling(5, min_periods=1).mean()
        df_metrics['rolling_avg_steps_10'] = df_metrics['num_steps'].rolling(10, min_periods=1).mean()
        df_metrics['cumulative_reward'] = df_metrics['total_reward'].cumsum()

        # cumulative/rolling metrics with only successful iterations
        df_metrics['rolling_avg_steps_success_3'] = self.rolling_avg_steps_success(df_metrics, window=3)
        df_metrics['rolling_avg_steps_success_5'] = self.rolling_avg_steps_success(df_metrics, window=5)
        df_metrics['rolling_avg_steps_success_10'] = self.rolling_avg_steps_success(df_metrics, window=10)
    
        
        self.metrics = df_metrics
    
    @staticmethod
    def aggregate_metrics(metric_list, name):
        metrics = pd.concat([m.metrics for m in metric_list], ignore_index=True)
        aggregated = {
            'benchmark_name': name,
            'total_episodes': len(metrics),
            'overall_success_rate': metrics['success'].mean(),
            'avg_steps': metrics['num_steps'].mean(),
            'avg_steps_successful': metrics[metrics['success'] == 1]['num_steps'].mean(),
            'avg_total_reward': metrics['total_reward'].mean(),
            'avg_reward_per_step': metrics['reward_per_step'].mean(),
            'termination_counts': metrics['termination_type'].value_counts().to_dict(),
        }
        return aggregated
    
    @staticmethod
    def aggregate_metrics_perepisode(metric_list, name):
        numeric_cols = [
            'success', 'num_steps', 'steps_efficiency',
            'total_reward', 'final_reward', 'reward_per_step',
            'cumulative_success_rate',
            'rolling_success_rate_3', 'rolling_success_rate_5', 'rolling_success_rate_10',
            'rolling_avg_steps_3', 'rolling_avg_steps_5', 'rolling_avg_steps_10',
            'cumulative_reward',
            'rolling_avg_steps_success_3', 'rolling_avg_steps_success_5', 'rolling_avg_steps_success_10',
        ]

        all_dfs = [m.metrics.set_index('iteration') for m in metric_list]
        existing_cols = [c for c in numeric_cols if all(c in df.columns for df in all_dfs)]

        combined = pd.concat([df[existing_cols] for df in all_dfs], keys=range(len(all_dfs)))
        avg = combined.groupby(level=1).mean()
        avg.index.name = 'iteration'
        avg = avg.reset_index()

        steps_stack = pd.concat([df['num_steps'] for df in all_dfs], axis=1)
        avg['num_steps_std'] = steps_stack.std(axis=1).values

        return {
            'benchmark_name': name,
            'metrics_df': avg,
        }

    def append_missing_iterations(self, metrics_list, current_counter, target_counter):  
        while current_counter < target_counter:
            metrics = {
                'iteration': current_counter,
                
                # SUCCESS METRICS
                'success': 0,
                'reached_goal': False,
                
                # EFFICIENCY METRICS
                'num_steps': 0,
                'steps_efficiency': 0.0,
                
                # REWARD METRICS
                'total_reward': 0.0,
                'final_reward': 0.0,
                'reward_per_step': 0.0,
                
                # TERMINATION METRICS
                'termination_type': 'error',
            }
            metrics_list.append(metrics)
            current_counter += 1

        return current_counter

    
    def is_successful(self, iter):
        terminal = iter[iter['is_terminated'] == True]
        if len(terminal) > 0:
            return int(terminal.iloc[-1]['reward'] == 1)
        return 0
    def reached_goal(self, iter):
        return bool(self.is_successful(iter))
    
    def num_steps(self, iter):
        return iter['step'].max() + 1
    
    def steps_efficiency(self, iter):
        num_steps = self.num_steps(iter)
        return 1.0 / num_steps if num_steps > 0 else 0.0
    
    def total_reward(self, iter):
        return iter['reward'].sum() 
    def final_reward(self, iter):
        return iter.iloc[-1]['reward']
    def reward_per_step(self, iter):
        total_reward = self.total_reward(iter)
        num_steps = self.num_steps(iter)
        return total_reward / num_steps if num_steps > 0 else 0.0
    
    def termination_type(self, iter): # goal, hole, max_steps_exceeded, error
        terminal = iter[iter['is_terminated'] == True]
        if len(terminal) > 0:
            return 'goal' if terminal.iloc[-1]['reward'] == 1 else 'hole'
        return 'max_steps_exceeded'
    
    def rolling_avg_steps_success(self, df_metrics, window=3):
        rolling_avg = []
    
        for i in range(len(df_metrics)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1

            window_data = df_metrics.iloc[start_idx:end_idx]

            successful_in_window = window_data[window_data['success'] == 1]

            if len(successful_in_window) > 0:
                avg_steps = successful_in_window['num_steps'].mean()
                rolling_avg.append(avg_steps)
            else:
                rolling_avg.append(np.nan)

        return rolling_avg