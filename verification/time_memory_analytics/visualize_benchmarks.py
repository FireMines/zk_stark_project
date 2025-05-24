import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set style for publication-quality plots
# Use a more compatible approach for different matplotlib versions
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            # Fallback to a basic style
            plt.style.use('default')
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'

# Set color palette - use a more compatible approach
try:
    sns.set_palette("husl")
except:
    pass

# Read all CSV files
csv_files = ['unified_metrics_20250515_165005.csv',  # batch size 2
             'unified_metrics_20250515_165048.csv',  # batch size 4
             'unified_metrics_20250515_165144.csv',  # batch size 5
             'unified_metrics_20250515_165250.csv',  # batch size 6
             'unified_metrics_20250515_165403.csv',  # batch size 7
             'unified_metrics_20250515_165525.csv',  # batch size 8
             'unified_metrics_20250515_165659.csv',  # batch size 9
             'unified_metrics_20250515_165950.csv',  # batch size 15
             'unified_metrics_20250515_170256.csv',  # batch size 16
             'unified_metrics_20250515_170621.csv']  # batch size 17
             
# Load and combine all data
all_data = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except FileNotFoundError:
        print(f"Warning: {file} not found, skipping...")

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
else:
    print("No data files found!")
    exit()

# Calculate statistics for each batch size
def calculate_stats(df, metric):
    """Calculate mean, std, and CV for a metric grouped by batch size"""
    stats = df.groupby('batch')[metric].agg(['mean', 'std', 'count'])
    stats['cv'] = (stats['std'] / stats['mean']) * 100
    return stats

# Color schemes - Make them more distinct
snark_color = '#2E7D32'  # Darker green
stark_color = '#C62828'  # Darker red
comparison_colors = [snark_color, stark_color]

# Create output directory
output_dir = Path('benchmark_graphs6')
output_dir.mkdir(exist_ok=True)

# 1. TIME COMPARISON GRAPHS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Execution Time Comparison: zkSNARK vs zkSTARK', fontsize=16, fontweight='bold')

# Aggregate time metrics
snark_time_cols = ['snark_t_compile', 'snark_t_setup', 'snark_t_proof', 'snark_t_export']
stark_time_cols = ['stark_t_build', 'stark_t_setup', 'stark_t_proof']

# Calculate total times (excluding witness)
combined_df['snark_total_time'] = combined_df[snark_time_cols].sum(axis=1)
combined_df['stark_total_time'] = combined_df[stark_time_cols].sum(axis=1)

# Total time comparison
ax = axes[0, 0]
time_data = combined_df.groupby('batch')[['snark_total_time', 'stark_total_time']].mean()
time_data.plot(kind='bar', ax=ax, color=comparison_colors, width=0.8)
ax.set_title('Total Execution Time by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(['zkSNARK', 'zkSTARK'])

# Individual phases - SNARK
ax = axes[0, 1]
snark_phases = combined_df.groupby('batch')[snark_time_cols].mean()
snark_phases.plot(kind='bar', ax=ax, color=sns.color_palette("Blues", len(snark_time_cols)))
ax.set_title('zkSNARK Individual Phase Times')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(['Compile', 'Setup', 'Proof', 'Export'])

# Individual phases - STARK
ax = axes[1, 0]
stark_phases = combined_df.groupby('batch')[stark_time_cols].mean()
stark_phases.plot(kind='bar', ax=ax, color=sns.color_palette("Reds", len(stark_time_cols)))
ax.set_title('zkSTARK Individual Phase Times')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(['Build', 'Setup', 'Proof'])

# Speedup analysis
ax = axes[1, 1]
speedup_data = time_data['snark_total_time'] / time_data['stark_total_time']
speedup_data.plot(kind='bar', ax=ax, color='green', alpha=0.7)
ax.set_title('Speedup Factor (SNARK/STARK)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Speedup Factor (x)')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
# Save in multiple formats for LaTeX compatibility
plt.savefig(output_dir / 'time_comparison.pdf', bbox_inches='tight')  # PDF for LaTeX
plt.savefig(output_dir / 'time_comparison.svg', bbox_inches='tight')  # SVG backup
plt.savefig(output_dir / 'time_comparison.png', dpi=300, bbox_inches='tight')  # PNG backup
plt.close()

# 2. MEMORY USAGE COMPARISON
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Memory Usage Comparison: zkSNARK vs zkSTARK', fontsize=16, fontweight='bold')

# Memory metrics
snark_mem_cols = ['snark_mem_compile', 'snark_mem_setup', 'snark_mem_proof', 'snark_mem_export']
stark_mem_cols = ['stark_mem_build', 'stark_mem_setup', 'stark_mem_proof']

# Peak memory comparison
ax = axes[0, 0]
combined_df['snark_peak_mem'] = combined_df[snark_mem_cols].max(axis=1)
combined_df['stark_peak_mem'] = combined_df[stark_mem_cols].max(axis=1)
mem_data = combined_df.groupby('batch')[['snark_peak_mem', 'stark_peak_mem']].mean()
mem_data.plot(kind='bar', ax=ax, color=comparison_colors, width=0.8)
ax.set_title('Peak Memory Usage by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(['zkSNARK', 'zkSTARK'])

# Memory phases - SNARK
ax = axes[0, 1]
snark_mem_phases = combined_df.groupby('batch')[snark_mem_cols].mean()
snark_mem_phases.plot(kind='bar', ax=ax, color=sns.color_palette("Blues", len(snark_mem_cols)))
ax.set_title('zkSNARK Memory Usage by Phase')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(['Compile', 'Setup', 'Proof', 'Export'])

# Memory phases - STARK
ax = axes[1, 0]
stark_mem_phases = combined_df.groupby('batch')[stark_mem_cols].mean()
stark_mem_phases.plot(kind='bar', ax=ax, color=sns.color_palette("Reds", len(stark_mem_cols)))
ax.set_title('zkSTARK Memory Usage by Phase')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(['Build', 'Setup', 'Proof'])

# Memory efficiency ratio
ax = axes[1, 1]
mem_ratio = mem_data['snark_peak_mem'] / mem_data['stark_peak_mem']
mem_ratio.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
ax.set_title('Memory Efficiency Ratio (SNARK/STARK)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Ratio')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / 'memory_comparison.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'memory_comparison.svg', bbox_inches='tight')
plt.savefig(output_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PROOF SIZE COMPARISON
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Proof Size Comparison', fontsize=16, fontweight='bold')

# Proof sizes - LOG SCALE due to huge difference
ax = axes[0, 0]
proof_data = combined_df.groupby('batch')[['snark_proof_size', 'stark_proof_size']].mean()
# Convert to KB for readability
proof_data_kb = proof_data / 1024
proof_data_kb.plot(kind='bar', ax=ax, color=comparison_colors, width=0.8, logy=True)
ax.set_title('Average Proof Size by Batch Size (Log Scale)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (KB, log scale)')
ax.legend(['zkSNARK', 'zkSTARK'])
ax.grid(True, alpha=0.3)

# Separate visualizations for SNARK and STARK proof sizes
ax = axes[0, 1]
proof_data_kb['snark_proof_size'].plot(kind='bar', ax=ax, color=snark_color, width=0.6)
ax.set_title('zkSNARK Proof Size by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (KB)')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
proof_data_kb['stark_proof_size'].plot(kind='bar', ax=ax, color=stark_color, width=0.6)
ax.set_title('zkSTARK Proof Size by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (KB)')
ax.grid(True, alpha=0.3)

# Proof size ratio
ax = axes[1, 1]
size_ratio = proof_data['stark_proof_size'] / proof_data['snark_proof_size']
size_ratio.plot(kind='bar', ax=ax, color='orange', alpha=0.7)
ax.set_title('Proof Size Ratio (STARK/SNARK)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Size Ratio (x)')
# Add value labels on bars
for i, v in enumerate(size_ratio):
    ax.text(i, v + 10, f'{v:.0f}x', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'proof_size_comparison.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'proof_size_comparison.svg', bbox_inches='tight')
plt.savefig(output_dir / 'proof_size_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. STATISTICAL ANALYSIS - COEFFICIENT OF VARIATION
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Coefficient of Variation Analysis', fontsize=16, fontweight='bold')

# Time CV
ax = axes[0, 0]
time_metrics = ['snark_total_time', 'stark_total_time']
cv_time = pd.DataFrame({
    metric: calculate_stats(combined_df, metric)['cv'] 
    for metric in time_metrics
})
cv_time.plot(kind='bar', ax=ax, color=comparison_colors, width=0.8)
ax.set_title('Time Measurements - Coefficient of Variation')
ax.set_xlabel('Batch Size')
ax.set_ylabel('CV (%)')
ax.legend(['zkSNARK', 'zkSTARK'])

# Memory CV
ax = axes[0, 1]
mem_metrics = ['snark_peak_mem', 'stark_peak_mem']
cv_mem = pd.DataFrame({
    metric: calculate_stats(combined_df, metric)['cv'] 
    for metric in mem_metrics
})
cv_mem.plot(kind='bar', ax=ax, color=comparison_colors, width=0.8)
ax.set_title('Memory Usage - Coefficient of Variation')
ax.set_xlabel('Batch Size')
ax.set_ylabel('CV (%)')
ax.legend(['zkSNARK', 'zkSTARK'])

# Proof size CV - Make sure both are visible
ax = axes[1, 0]
proof_metrics = ['snark_proof_size', 'stark_proof_size']
cv_proof = pd.DataFrame({
    metric: calculate_stats(combined_df, metric)['cv'] 
    for metric in proof_metrics
})

# Debug: Print CV values to understand the issue
print("CV values for proof sizes:")
print(cv_proof)
print("SNARK proof size statistics:")
print(combined_df['snark_proof_size'].describe())

# Ensure both are plotted with distinct colors and better formatting
bar_width = 0.35
x_pos = np.arange(len(cv_proof.index))
rects1 = ax.bar(x_pos - bar_width/2, cv_proof['snark_proof_size'], bar_width, 
                color=snark_color, alpha=0.8, label='zkSNARK')
rects2 = ax.bar(x_pos + bar_width/2, cv_proof['stark_proof_size'], bar_width, 
                color=stark_color, alpha=0.8, label='zkSTARK')
ax.set_title('Proof Size - Coefficient of Variation')
ax.set_xlabel('Batch Size')
ax.set_ylabel('CV (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(cv_proof.index)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars for clarity
for rect, value in zip(rects1, cv_proof['snark_proof_size']):
    if not np.isnan(value) and value > 0:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=9)
                
for rect, value in zip(rects2, cv_proof['stark_proof_size']):
    if not np.isnan(value) and value > 0:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=9)

# Combined CV visualization
ax = axes[1, 1]
cv_combined = pd.DataFrame({
    'Time': cv_time.mean(axis=1),
    'Memory': cv_mem.mean(axis=1),
    'Proof Size': cv_proof.mean(axis=1)
})
cv_combined.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=8)
ax.set_title('Average CV Across All Metrics')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Average CV (%)')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'cv_analysis.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'cv_analysis.svg', bbox_inches='tight')
plt.savefig(output_dir / 'cv_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. DETAILED STARK-ONLY ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('zkSTARK Detailed Analysis', fontsize=16, fontweight='bold')

# STARK time breakdown
ax = axes[0, 0]
stark_phases.plot(kind='area', ax=ax, alpha=0.7, color=sns.color_palette("Reds", len(stark_time_cols)))
ax.set_title('zkSTARK Time Distribution')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(['Build', 'Setup', 'Proof'])

# STARK memory breakdown
ax = axes[0, 1]
stark_mem_phases.plot(kind='area', ax=ax, alpha=0.7, color=sns.color_palette("Blues", len(stark_mem_cols)))
ax.set_title('zkSTARK Memory Distribution')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(['Build', 'Setup', 'Proof'])

# STARK scalability
ax = axes[1, 0]
stark_scaling = combined_df.groupby('batch')['stark_total_time'].mean()
batch_sizes = stark_scaling.index.values
time_values = stark_scaling.values
ax.scatter(batch_sizes, time_values, s=100, alpha=0.7, color='red')
z = np.polyfit(batch_sizes, time_values, 1)
p = np.poly1d(z)
ax.plot(batch_sizes, p(batch_sizes), "r--", alpha=0.8)
ax.set_title('zkSTARK Time Scalability')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Total Time (seconds)')

# STARK efficiency metrics
ax = axes[1, 1]
stark_data = combined_df.groupby('batch')[['stark_t_setup', 'stark_t_proof']].mean()
efficiency = stark_data['stark_t_proof'] / stark_data['stark_t_setup']
efficiency.plot(kind='bar', ax=ax, color='darkred', alpha=0.7)
ax.set_title('zkSTARK Proof/Setup Time Ratio')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Ratio')

plt.tight_layout()
plt.savefig(output_dir / 'stark_detailed.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'stark_detailed.svg', bbox_inches='tight')
plt.savefig(output_dir / 'stark_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. DETAILED SNARK-ONLY ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('zkSNARK Detailed Analysis', fontsize=16, fontweight='bold')

# SNARK time breakdown
ax = axes[0, 0]
snark_phases.plot(kind='area', ax=ax, alpha=0.7, color=sns.color_palette("Blues", len(snark_time_cols)))
ax.set_title('zkSNARK Time Distribution')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(['Compile', 'Setup', 'Proof', 'Export'])

# SNARK memory breakdown
ax = axes[0, 1]
snark_mem_phases.plot(kind='area', ax=ax, alpha=0.7, color=sns.color_palette("Greens", len(snark_mem_cols)))
ax.set_title('zkSNARK Memory Distribution')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(['Compile', 'Setup', 'Proof', 'Export'])

# SNARK scalability
ax = axes[1, 0]
snark_scaling = combined_df.groupby('batch')['snark_total_time'].mean()
batch_sizes = snark_scaling.index.values
time_values = snark_scaling.values
ax.scatter(batch_sizes, time_values, s=100, alpha=0.7, color='blue')
z = np.polyfit(batch_sizes, time_values, 1)
p = np.poly1d(z)
ax.plot(batch_sizes, p(batch_sizes), "b--", alpha=0.8)
ax.set_title('zkSNARK Time Scalability')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Total Time (seconds)')

# SNARK efficiency metrics
ax = axes[1, 1]
snark_data = combined_df.groupby('batch')[['snark_t_setup', 'snark_t_proof']].mean()
efficiency = snark_data['snark_t_proof'] / snark_data['snark_t_setup']
efficiency.plot(kind='bar', ax=ax, color='darkblue', alpha=0.7)
ax.set_title('zkSNARK Proof/Setup Time Ratio')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Ratio')

plt.tight_layout()
plt.savefig(output_dir / 'snark_detailed.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'snark_detailed.svg', bbox_inches='tight')
plt.savefig(output_dir / 'snark_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. DISTRIBUTION ANALYSIS (Violin Plots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Distribution Analysis', fontsize=16, fontweight='bold')

# Time distribution - with better colors
ax = axes[0, 0]
melted_time = pd.melt(combined_df, id_vars=['batch'], 
                      value_vars=['snark_total_time', 'stark_total_time'],
                      var_name='System', value_name='Time')
melted_time['System'] = melted_time['System'].map({'snark_total_time': 'zkSNARK', 'stark_total_time': 'zkSTARK'})
# Use explicit color palette
colors = {'zkSNARK': snark_color, 'zkSTARK': stark_color}
sns.violinplot(data=melted_time, x='batch', y='Time', hue='System', ax=ax, palette=colors, alpha=0.8)
ax.set_title('Time Distribution by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (seconds)')
ax.legend(title='System', frameon=True, fancybox=True, shadow=True)

# Memory distribution - with better colors
ax = axes[0, 1]
melted_mem = pd.melt(combined_df, id_vars=['batch'], 
                     value_vars=['snark_peak_mem', 'stark_peak_mem'],
                     var_name='System', value_name='Memory')
melted_mem['System'] = melted_mem['System'].map({'snark_peak_mem': 'zkSNARK', 'stark_peak_mem': 'zkSTARK'})
sns.violinplot(data=melted_mem, x='batch', y='Memory', hue='System', ax=ax, palette=colors, alpha=0.8)
ax.set_title('Memory Distribution by Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory (MB)')
ax.legend(title='System', frameon=True, fancybox=True, shadow=True)

# Proof size distribution - USE LOG SCALE
ax = axes[1, 0]
melted_proof = pd.melt(combined_df, id_vars=['batch'], 
                       value_vars=['snark_proof_size', 'stark_proof_size'],
                       var_name='System', value_name='Proof Size')
melted_proof['System'] = melted_proof['System'].map({'snark_proof_size': 'zkSNARK', 'stark_proof_size': 'zkSTARK'})
# Don't convert to KB to preserve the full difference
sns.violinplot(data=melted_proof, x='batch', y='Proof Size', hue='System', ax=ax, palette=colors, alpha=0.8)
ax.set_yscale('log')  # Use log scale to show both SNARK and STARK
ax.set_title('Proof Size Distribution by Batch Size (Log Scale)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (bytes, log scale)')
ax.legend(title='System', frameon=True, fancybox=True, shadow=True)

# Combined efficiency violin plot
ax = axes[1, 1]
combined_df['time_efficiency'] = combined_df['stark_total_time'] / combined_df['snark_total_time']
combined_df['mem_efficiency'] = combined_df['stark_peak_mem'] / combined_df['snark_peak_mem']
melted_eff = pd.melt(combined_df, id_vars=['batch'], 
                     value_vars=['time_efficiency', 'mem_efficiency'],
                     var_name='Metric', value_name='Efficiency Ratio')
melted_eff['Metric'] = melted_eff['Metric'].map({'time_efficiency': 'Time', 'mem_efficiency': 'Memory'})
eff_colors = {'Time': '#1f77b4', 'Memory': '#ff7f0e'}
sns.violinplot(data=melted_eff, x='batch', y='Efficiency Ratio', hue='Metric', ax=ax, palette=eff_colors, alpha=0.8)
ax.set_title('Efficiency Ratios (STARK/SNARK)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Ratio')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.legend(title='Metric', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / 'distribution_analysis.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'distribution_analysis.svg', bbox_inches='tight')
plt.savefig(output_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. HEATMAP ANALYSIS
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Performance Metrics Heatmap', fontsize=16, fontweight='bold')

# Create correlation matrix for SNARK metrics
snark_metrics = ['snark_t_compile', 'snark_t_setup', 'snark_t_proof', 'snark_t_export',
                 'snark_mem_compile', 'snark_mem_setup', 'snark_mem_proof', 'snark_mem_export',
                 'snark_proof_size']
snark_corr = combined_df[snark_metrics].corr()
sns.heatmap(snark_corr, annot=True, cmap='Blues', center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('zkSNARK Metrics Correlation')

# Create correlation matrix for STARK metrics
stark_metrics = ['stark_t_build', 'stark_t_setup', 'stark_t_proof',
                 'stark_mem_build', 'stark_mem_setup', 'stark_mem_proof',
                 'stark_proof_size']
stark_corr = combined_df[stark_metrics].corr()
sns.heatmap(stark_corr, annot=True, cmap='Reds', center=0, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('zkSTARK Metrics Correlation')

plt.tight_layout()
plt.savefig(output_dir / 'correlation_heatmap.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'correlation_heatmap.svg', bbox_inches='tight')
plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. SUMMARY STATISTICS TABLE
def create_summary_table():
    summary_data = []
    
    for batch in sorted(combined_df['batch'].unique()):
        batch_data = combined_df[combined_df['batch'] == batch]
        
        # Calculate statistics for each metric
        metrics = {
            'SNARK Total Time': batch_data['snark_total_time'],
            'STARK Total Time': batch_data['stark_total_time'],
            'SNARK Peak Memory': batch_data['snark_peak_mem'],
            'STARK Peak Memory': batch_data['stark_peak_mem'],
            'SNARK Proof Size': batch_data['snark_proof_size'] / 1024,  # KB
            'STARK Proof Size': batch_data['stark_proof_size'] / 1024,  # KB
        }
        
        for metric_name, values in metrics.items():
            summary_data.append({
                'Batch Size': batch,
                'Metric': metric_name,
                'Mean': values.mean(),
                'Std Dev': values.std(),
                'CV (%)': (values.std() / values.mean()) * 100 if values.mean() != 0 else 0,
                'Min': values.min(),
                'Max': values.max()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    return summary_df

summary_table = create_summary_table()

# 10. SCALING TRENDS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Scaling Trends Analysis', fontsize=16, fontweight='bold')

# Time scaling
ax = axes[0, 0]
batch_sizes = sorted(combined_df['batch'].unique())
snark_times = [combined_df[combined_df['batch'] == b]['snark_total_time'].mean() for b in batch_sizes]
stark_times = [combined_df[combined_df['batch'] == b]['stark_total_time'].mean() for b in batch_sizes]

ax.loglog(batch_sizes, snark_times, 'o-', label='zkSNARK', color=snark_color, linewidth=2, markersize=8)
ax.loglog(batch_sizes, stark_times, 's-', label='zkSTARK', color=stark_color, linewidth=2, markersize=8)
ax.set_xlabel('Batch Size (log scale)')
ax.set_ylabel('Time (seconds, log scale)')
ax.set_title('Time Scaling (Log-Log)')
ax.legend()
ax.grid(True, alpha=0.3)

# Memory scaling
ax = axes[0, 1]
snark_mems = [combined_df[combined_df['batch'] == b]['snark_peak_mem'].mean() for b in batch_sizes]
stark_mems = [combined_df[combined_df['batch'] == b]['stark_peak_mem'].mean() for b in batch_sizes]

ax.loglog(batch_sizes, snark_mems, 'o-', label='zkSNARK', color=snark_color, linewidth=2, markersize=8)
ax.loglog(batch_sizes, stark_mems, 's-', label='zkSTARK', color=stark_color, linewidth=2, markersize=8)
ax.set_xlabel('Batch Size (log scale)')
ax.set_ylabel('Memory (MB, log scale)')
ax.set_title('Memory Scaling (Log-Log)')
ax.legend()
ax.grid(True, alpha=0.3)

# Proof size scaling
ax = axes[1, 0]
snark_proofs = [combined_df[combined_df['batch'] == b]['snark_proof_size'].mean() / 1024 for b in batch_sizes]
stark_proofs = [combined_df[combined_df['batch'] == b]['stark_proof_size'].mean() / 1024 for b in batch_sizes]

ax.semilogy(batch_sizes, snark_proofs, 'o-', label='zkSNARK', color=snark_color, linewidth=2, markersize=8)
ax.semilogy(batch_sizes, stark_proofs, 's-', label='zkSTARK', color=stark_color, linewidth=2, markersize=8)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (KB, log scale)')
ax.set_title('Proof Size Scaling')
ax.legend()
ax.grid(True, alpha=0.3)

# Efficiency trends
ax = axes[1, 1]
time_ratios = [s/n for s, n in zip(stark_times, snark_times)]
mem_ratios = [s/n for s, n in zip(stark_mems, snark_mems)]

ax.plot(batch_sizes, time_ratios, 'o-', label='Time Ratio (STARK/SNARK)', linewidth=2, markersize=8)
ax.plot(batch_sizes, mem_ratios, 's-', label='Memory Ratio (STARK/SNARK)', linewidth=2, markersize=8)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Ratio')
ax.set_title('Efficiency Ratios Trend')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / 'scaling_trends.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'scaling_trends.svg', bbox_inches='tight')
plt.savefig(output_dir / 'scaling_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All graphs have been generated and saved in the '{output_dir}' directory!")
print("\nGenerated visualizations in multiple formats:")
print("1. time_comparison.pdf/.svg/.png - Comprehensive time analysis")
print("2. memory_comparison.pdf/.svg/.png - Memory usage comparison")
print("3. proof_size_comparison.pdf/.svg/.png - Proof size analysis")
print("4. cv_analysis.pdf/.svg/.png - Coefficient of variation analysis")
print("5. stark_detailed.pdf/.svg/.png - Detailed zkSTARK analysis")
print("6. snark_detailed.pdf/.svg/.png - Detailed zkSNARK analysis")
print("7. distribution_analysis.pdf/.svg/.png - Performance distributions")
print("8. correlation_heatmap.pdf/.svg/.png - Correlation analysis")
print("9. scaling_trends.pdf/.svg/.png - Scaling behavior analysis")
print("10. summary_statistics.csv - Statistical summary table")
print("\n📄 For LaTeX: Use the .pdf files for best quality!")
print("🔧 SVG files can be converted to other formats if needed")
print("🖼️  PNG files are provided as backup (300 DPI)")