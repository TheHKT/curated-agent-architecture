import matplotlib.pyplot as plt

class Visualizer: #TODO: rename iteration to episode

    def print_metrics_summary(df_metrics, benchmark_name):
        fig, axes = plt.subplots(4, 2, figsize=(15, 10))
        fig.suptitle(f'{benchmark_name}', fontsize=16, fontweight='bold')

        default_color = "cyan"
        
        # 1. Success Rate Over Time
        axes[0, 0].plot(df_metrics['iteration'], df_metrics['cumulative_success_rate'], marker='o', color=default_color)
        axes[0, 0].set_title('Cumulative Success Rate')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)


        # 2. Cumulative Reward
        axes[0, 1].plot(df_metrics['iteration'], df_metrics['cumulative_reward'], marker='o', color=default_color)
        axes[0, 1].set_title('Cumulative Reward')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].grid(True, alpha=0.3)


        # 3. Number of Steps
        axes[1, 0].plot(
            df_metrics['iteration'],
            df_metrics['num_steps'],
            color=default_color,
            linewidth=1.5,
            alpha=0.8
        )
        success_mask = df_metrics['success'] == 1
        fail_mask = ~success_mask
        axes[1, 0].scatter(
            df_metrics.loc[success_mask, 'iteration'],
            df_metrics.loc[success_mask, 'num_steps'],
            color='green',
            label='Success',
            s=35
        )
        axes[1, 0].scatter(
            df_metrics.loc[fail_mask, 'iteration'],
            df_metrics.loc[fail_mask, 'num_steps'],
            color='red',
            label='Failure',
            s=35
        )
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Steps')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='best')


        # 4. Steps Efficiency
        axes[1, 1].plot(
            df_metrics['iteration'],
            df_metrics['steps_efficiency'],
            color=default_color,
            linewidth=1.5,
            alpha=0.8
        )
        success_mask = df_metrics['success'] == 1
        fail_mask = ~success_mask
        axes[1, 1].scatter(
            df_metrics.loc[success_mask, 'iteration'],
            df_metrics.loc[success_mask, 'steps_efficiency'],
            color='green',
            label='Success',
            s=35
        )
        axes[1, 1].scatter(
            df_metrics.loc[fail_mask, 'iteration'],
            df_metrics.loc[fail_mask, 'steps_efficiency'],
            color='red',
            label='Failure',
            s=35
        )
        axes[1, 1].set_title('Steps Efficiency')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='best')
        

        # 5. Rolling Average Steps
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_3'], label='Window=3', marker='o')
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_5'], label='Window=5', marker='s')
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_10'], label='Window=10', marker='^')
        axes[2, 0].set_title('Rolling Average Steps of Successful Episodes')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Average Steps')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)


        # 6. Roling Success Rate
        axes[2, 1].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_3'], label='Window=3', marker='o')
        axes[2, 1].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_5'], label='Window=5', marker='s')
        axes[2, 1].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_10'], label='Window=10', marker='^')
        axes[2, 1].set_title('Rolling Success Rate')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Success Rate')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)


        # 7. Termination Type Distribution
        termination_counts = df_metrics['termination_type'].value_counts()
        color_map = {
            'goal': 'green',
            'error': 'red',
            'max_steps_exceeded': 'orange',
            'hole': 'darkred'
        }
        colors = [color_map.get(term, 'gray') for term in termination_counts.index]
        axes[3, 0].bar(termination_counts.index, termination_counts.values, color=colors)
        axes[3, 0].set_title('Termination Type Distribution')
        axes[3, 0].set_ylabel('Count')
        axes[3, 0].grid(True, alpha=0.3, axis='y')


        #8. Unused
        axes[3, 1].axis('off')

        plt.tight_layout()
        plt.show()