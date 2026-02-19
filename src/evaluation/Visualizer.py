import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Visualizer:
    def print_aggregated_metrics_perepisode(aggregated_metrics: list, title, save_path=None):
        """
        aggregated_metrics: list of dicts from Metrics.aggregate_metrics_perepisode()
        Each dict has: { 'benchmark_name': str, 'metrics_df': DataFrame }
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        colors  = plt.cm.tab10(np.linspace(0, 0.5, len(aggregated_metrics)))
        markers = ['o', 's', '^', 'D', 'v']

        def plot_lines(ax, col, ylabel, plot_title, ylim=None):
            for i, m in enumerate(aggregated_metrics):
                avg = m['metrics_df']
                if col not in avg.columns:
                    continue
                ax.plot(avg['iteration'], avg[col],
                        label=m['benchmark_name'], color=colors[i],
                        marker=markers[i % len(markers)], markersize=3, linewidth=1.5)
            ax.set_title(plot_title)
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # --- 1. Cumulative Success Rate ---
        plot_lines(axes[0, 0], 'cumulative_success_rate', 'Success Rate',
                   'Cumulative Success Rate (avg across runs)', ylim=(0, 1.05))

        # --- 2. Cumulative Reward ---
        plot_lines(axes[0, 1], 'cumulative_reward', 'Total Reward',
                   'Cumulative Reward (avg across runs)')

        # --- 3. Steps per Episode with std shading ---
        ax = axes[1, 0]
        for i, m in enumerate(aggregated_metrics):
            avg = m['metrics_df']
            iters = avg['iteration']
            ax.plot(iters, avg['num_steps'],
                    label=m['benchmark_name'], color=colors[i],
                    marker=markers[i % len(markers)], markersize=3, linewidth=1.5)
            if 'num_steps_std' in avg.columns:
                ax.fill_between(iters,
                                avg['num_steps'] - avg['num_steps_std'],
                                avg['num_steps'] + avg['num_steps_std'],
                                alpha=0.15, color=colors[i])
        ax.set_title('Avg Steps per Episode (+/-1 std)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 4. Rolling Avg Steps of Successful Episodes (window=10) ---
        plot_lines(axes[1, 1], 'rolling_avg_steps_success_10', 'Average Steps',
                   'Rolling Avg Steps - Successful Episodes (Window=10)')

        # --- 5. Rolling Success Rate (window=10) ---
        plot_lines(axes[2, 0], 'rolling_success_rate_10', 'Success Rate',
                   'Rolling Success Rate (Window=10)', ylim=(0, 1.05))

        # --- 6. Avg Reward per Step ---
        plot_lines(axes[2, 1], 'reward_per_step', 'Reward / Step',
                   'Avg Reward per Step (avg across runs)')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_aggregated_metrics_summary(aggregated_metrics: list, title, save_path=None):
        # Extract benchmark names once
        benchmark_names = [m['benchmark_name'] for m in aggregated_metrics]

        # Package the data into a list of tuples to avoid repetitive code
        # Format: (Title, Data List, Y-Axis Label, String Format)
        plots_data = [
            ("Overall Success Rate", [m['overall_success_rate'] for m in aggregated_metrics], "Success Rate", "%.2f"),
            #("Average Steps (All Episodes)", [m['avg_steps'] for m in aggregated_metrics], "Number of Steps", "%.1f"),
            ("Average Steps (Successful)", [m['avg_steps_successful'] for m in aggregated_metrics], "Number of Steps", "%.1f"),
            #("Average Total Reward", [m['avg_total_reward'] for m in aggregated_metrics], "Total Reward", "%.2f"),
            #("Average Reward Per Step", [m['avg_reward_per_step'] for m in aggregated_metrics], "Reward Per Step", "%.2f"),
            ("Total Episodes", [m['total_episodes'] for m in aggregated_metrics], "Episode Count", "%d")
        ]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        # Styling variables
        colors = plt.cm.Set3(np.linspace(0, 1, len(benchmark_names)))
        x_pos = np.arange(len(benchmark_names))
        bar_width = 0.6

        # Flatten axes for easy iteration
        axes = axes.flatten()

        for ax, (title, data, ylabel, fmt) in zip(axes, plots_data):
            # Added a subtle edgecolor for better definition
            bars = ax.bar(x_pos, data, bar_width, color=colors, edgecolor='dimgray', linewidth=0.5)

            # Labels and Titles
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=10)

            # Gridlines (set_axisbelow ensures grid is drawn *behind* the bars)
            ax.set_axisbelow(True) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            # Remove top and right borders (spines) for a cleaner, modern look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Dynamically give headroom so labels aren't cut off
            if "Success Rate" in title:
                ax.set_ylim(0, 1.15) # Visual headroom up to 1.15 for the text
            else:
                ax.margins(y=0.2) # Adds 20% dynamic headroom above the tallest bar

            # Use Matplotlib's built-in bar_label instead of manual text offsets
            ax.bar_label(bars, fmt=fmt, padding=4, fontsize=10)

        # h_pad and w_pad specifically fix the overlapping text between rows
        plt.tight_layout(h_pad=3.0, w_pad=2.0)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_metrics_summary(df_metrics, benchmark_name, save_path=None):
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
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
        #axes[1, 1].plot(
        #    df_metrics['iteration'],
        #    df_metrics['steps_efficiency'],
        #    color=default_color,
        #    linewidth=1.5,
        #    alpha=0.8
        #)
        #success_mask = df_metrics['success'] == 1
        #fail_mask = ~success_mask
        #axes[1, 1].scatter(
        #    df_metrics.loc[success_mask, 'iteration'],
        #    df_metrics.loc[success_mask, 'steps_efficiency'],
        #    color='green',
        #    label='Success',
        #    s=35
        #)
        #axes[1, 1].scatter(
        #    df_metrics.loc[fail_mask, 'iteration'],
        #    df_metrics.loc[fail_mask, 'steps_efficiency'],
        #    color='red',
        #    label='Failure',
        #    s=35
        #)
        #axes[1, 1].set_title('Steps Efficiency')
        #axes[1, 1].set_xlabel('Episode')
        #axes[1, 1].set_ylabel('Efficiency')
        #axes[1, 1].grid(True, alpha=0.3)
        #axes[1, 1].legend(loc='best')
        

        # 5. Rolling Average Steps
        axes[1, 1].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_3'], label='Window=3', marker='o')
        axes[1, 1].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_5'], label='Window=5', marker='s')
        axes[1, 1].plot(df_metrics['iteration'], df_metrics['rolling_avg_steps_success_10'], label='Window=10', marker='^')
        axes[1, 1].set_title('Rolling Average Steps of Successful Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Steps')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)


        # 6. Rolling Success Rate
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_3'], label='Window=3', marker='o')
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_5'], label='Window=5', marker='s')
        axes[2, 0].plot(df_metrics['iteration'], df_metrics['rolling_success_rate_10'], label='Window=10', marker='^')
        axes[2, 0].set_title('Rolling Success Rate')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Success Rate')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)


        # 7. Termination Type Distribution
        termination_counts = df_metrics['termination_type'].value_counts()
        color_map = {
            'goal': 'green',
            'error': 'red',
            'max_steps_exceeded': 'orange',
            'hole': 'darkred'
        }
        colors = [color_map.get(term, 'gray') for term in termination_counts.index]
        axes[2, 1].bar(termination_counts.index, termination_counts.values, color=colors)
        axes[2, 1].set_title('Termination Type Distribution')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].grid(True, alpha=0.3, axis='y')


        #8. Unused
        #axes[3, 1].axis('off')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()