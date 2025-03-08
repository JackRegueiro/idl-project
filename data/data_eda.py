import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import numpy as np

# Create visualization directory if it doesn't exist
VISUALIZATION_DIR = os.path.join(os.path.dirname(__file__), "eda_visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_copro_dataset(data_path: str) -> Dict[str, Any]:
    """
    Load the CoPro dataset from the specified path.
    
    Args:
        data_path: Path to the CoPro dataset JSON file or directory containing the file.
        
    Returns:
        The loaded dataset as a dictionary.
    """
    # Handle both directory and file paths
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "CoPro_v1.0.json")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CoPro dataset file not found at: {data_path}")
        
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding CoPro dataset JSON: {e}")
        raise
    except Exception as e:
        print(f"Error loading CoPro dataset: {e}")
        raise

def analyze_dataset(data: Dict[str, Any]) -> None:
    """
    Analyze the CoPro dataset and print statistics.
    
    Args:
        data: The loaded CoPro dataset.
    """
    # Extract training data
    train_data = data.get("ID_train_data", [])
    
    # Count total entries
    total_entries = len(train_data)
    print(f"Total number of entries: {total_entries}")
    
    # Count categories
    categories = [item.get("category") for item in train_data]
    category_counts = Counter(categories)
    print("\nCategory distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} entries ({count/total_entries*100:.2f}%)")
    
    # Count concepts
    concepts = [item.get("concept") for item in train_data]
    concept_counts = Counter(concepts)
    print(f"\nTotal number of unique concepts: {len(concept_counts)}")
    
    # Print top concepts
    print("\nTop 20 concepts:")
    for concept, count in concept_counts.most_common(20):
        print(f"  {concept}: {count} entries ({count/total_entries*100:.2f}%)")
    
    # Analyze concepts per category
    print("\nConcepts per category:")
    category_concepts = {}
    
    for category in category_counts:
        category_items = [item for item in train_data if item.get("category") == category]
        category_concepts[category] = [item.get("concept") for item in category_items]
        unique_concepts = set(category_concepts[category])
        print(f"  {category}: {len(unique_concepts)} unique concepts")
        
        # Print top concepts for this category
        concept_counts_in_category = Counter(category_concepts[category])
        print(f"    Top 5 concepts in {category}:")
        for concept, count in concept_counts_in_category.most_common(5):
            print(f"      {concept}: {count} entries ({count/len(category_items)*100:.2f}%)")
    
    # Create visualizations
    plot_category_distribution(category_counts)
    plot_top_concepts(concept_counts, 15)
    plot_concepts_by_category(category_concepts, top_n=5)
    
    # Focus on sexual category, as used in the paper
    if "sexual" in category_concepts:
        plot_sexual_category_concepts(category_concepts["sexual"])
        
    # Prompt length analysis
    analyze_prompt_lengths(train_data)

def plot_category_distribution(category_counts: Counter) -> None:
    """
    Plot the distribution of categories.
    
    Args:
        category_counts: Counter object containing category counts.
    """
    plt.figure(figsize=(12, 7))
    categories = [cat for cat, _ in category_counts.most_common()]
    counts = [count for _, count in category_counts.most_common()]
    percentages = [count / sum(counts) * 100 for count in counts]
    
    # Create bars with color gradient
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(categories)))
    bars = plt.bar(categories, counts, color=colors)
    
    # Add data labels on top of bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title('Distribution of Categories in CoPro Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Entries', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'category_distribution.png'), dpi=300)
    print(f"\nSaved category distribution plot to {os.path.join(VISUALIZATION_DIR, 'category_distribution.png')}")

def plot_top_concepts(concept_counts: Counter, top_n: int = 15) -> None:
    """
    Plot the distribution of the top N concepts.
    
    Args:
        concept_counts: Counter object containing concept counts.
        top_n: Number of top concepts to plot.
    """
    plt.figure(figsize=(14, 8))
    concepts = [concept for concept, _ in concept_counts.most_common(top_n)]
    counts = [count for _, count in concept_counts.most_common(top_n)]
    
    # Create horizontal bar chart with color gradient
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(concepts)))
    bars = plt.barh(concepts[::-1], counts[::-1], color=colors[::-1])
    
    # Add data labels inside bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width - width*0.05, bar.get_y() + bar.get_height()/2,
                f'{width}', ha='right', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.title(f'Top {top_n} Most Common Concepts in CoPro Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Entries', fontsize=12)
    plt.ylabel('Concept', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'top_concepts.png'), dpi=300)
    print(f"Saved top {top_n} concepts plot to {os.path.join(VISUALIZATION_DIR, 'top_concepts.png')}")

def plot_concepts_by_category(category_concepts: Dict[str, List[str]], top_n: int = 5) -> None:
    """
    Create a multi-panel plot showing top concepts for each category.
    
    Args:
        category_concepts: Dictionary mapping categories to lists of concepts.
        top_n: Number of top concepts to show for each category.
    """
    # Calculate number of rows and columns for subplots
    n_categories = len(category_concepts)
    n_cols = 2
    n_rows = (n_categories + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, (category, concepts) in enumerate(category_concepts.items()):
        if i < len(axes):
            ax = axes[i]
            concept_counts = Counter(concepts)
            top_concepts = [concept for concept, _ in concept_counts.most_common(top_n)]
            top_counts = [count for _, count in concept_counts.most_common(top_n)]
            
            # Create bars with color gradient
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_concepts)))
            bars = ax.barh(top_concepts[::-1], top_counts[::-1], color=colors[::-1])
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + width*0.05, bar.get_y() + bar.get_height()/2,
                       f'{width}', ha='left', va='center', fontsize=9)
            
            ax.set_title(f'{category} Category: Top {top_n} Concepts', fontsize=12)
            ax.set_xlabel('Number of Entries', fontsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'concepts_by_category.png'), dpi=300)
    print(f"Saved concepts by category plot to {os.path.join(VISUALIZATION_DIR, 'concepts_by_category.png')}")

def plot_sexual_category_concepts(concepts: List[str], top_n: int = 20) -> None:
    """
    Create a detailed visualization of concepts in the sexual category.
    
    Args:
        concepts: List of concepts in the sexual category.
        top_n: Number of top concepts to visualize.
    """
    concept_counts = Counter(concepts)
    
    plt.figure(figsize=(14, 10))
    
    # Get top concepts and their counts
    top_concepts = [concept for concept, _ in concept_counts.most_common(top_n)]
    top_counts = [count for _, count in concept_counts.most_common(top_n)]
    
    # Create horizontal bar chart with color gradient
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(top_concepts)))
    bars = plt.barh(top_concepts[::-1], top_counts[::-1], color=colors[::-1])
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + width*0.05, bar.get_y() + bar.get_height()/2,
               f'{width}', ha='left', va='center', fontsize=10)
    
    plt.title('Top Concepts in Sexual Category (Used in DES Paper)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Entries', fontsize=14)
    plt.ylabel('Concept', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'sexual_category_concepts.png'), dpi=300)
    print(f"Saved sexual category concepts plot to {os.path.join(VISUALIZATION_DIR, 'sexual_category_concepts.png')}")

def analyze_prompt_lengths(data: List[Dict[str, Any]]) -> None:
    """
    Analyze and visualize the lengths of safe and unsafe prompts.
    
    Args:
        data: List of data items containing safe and unsafe prompts.
    """
    # Extract prompt lengths
    safe_lengths = [len(item.get("safe_prompt", "")) for item in data]
    unsafe_lengths = [len(item.get("unsafe_prompt", "")) for item in data]
    
    plt.figure(figsize=(14, 6))
    
    # Histogram of prompt lengths
    plt.hist(safe_lengths, bins=50, alpha=0.6, label='Safe Prompts', color='green')
    plt.hist(unsafe_lengths, bins=50, alpha=0.6, label='Unsafe Prompts', color='red')
    
    # Add statistics to the plot
    plt.axvline(np.mean(safe_lengths), color='green', linestyle='dashed', linewidth=2, 
                label=f'Safe Avg: {np.mean(safe_lengths):.1f} chars')
    plt.axvline(np.mean(unsafe_lengths), color='red', linestyle='dashed', linewidth=2,
                label=f'Unsafe Avg: {np.mean(unsafe_lengths):.1f} chars')
    
    plt.title('Distribution of Prompt Lengths', fontsize=14, fontweight='bold')
    plt.xlabel('Prompt Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'prompt_length_distribution.png'), dpi=300)
    print(f"Saved prompt length distribution to {os.path.join(VISUALIZATION_DIR, 'prompt_length_distribution.png')}")
    
    # Print summary statistics
    print("\nPrompt Length Statistics:")
    print(f"  Safe prompts: Avg = {np.mean(safe_lengths):.1f}, Min = {min(safe_lengths)}, Max = {max(safe_lengths)}")
    print(f"  Unsafe prompts: Avg = {np.mean(unsafe_lengths):.1f}, Min = {min(unsafe_lengths)}, Max = {max(unsafe_lengths)}")

def main():
    # Default path to CoPro dataset
    data_path = os.path.join("..", "CoPro Dataset")
    
    try:
        # Load the dataset
        data = load_copro_dataset(data_path)
        
        # Analyze the dataset
        analyze_dataset(data)
        
        print(f"\nAll visualizations saved to: {VISUALIZATION_DIR}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 