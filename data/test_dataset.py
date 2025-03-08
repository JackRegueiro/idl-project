import os
import sys
import argparse
from data.dataset import CoProDataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the CoProDataset implementation')
    parser.add_argument('--data_path', type=str, default='CoPro Dataset', 
                        help='Path to the CoPro dataset directory or JSON file')
    parser.add_argument('--category', type=str, default='sexual',
                        help='Category to filter (default: sexual)')
    parser.add_argument('--num_examples', type=int, default=3,
                        help='Number of examples to display (default: 3)')
    args = parser.parse_args()
    
    # Resolve data path
    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist")
        sys.exit(1)
    
    print(f"Loading CoPro dataset from: {data_path}")
    print(f"Category: {args.category}")
    
    try:
        # Create dataset with the specified category
        dataset = CoProDataset(data_path, category=args.category)
        
        # Print dataset statistics
        print(f"\nDataset size: {len(dataset)}")
        
        # Show examples
        print("\nSample data points:")
        num_examples = min(args.num_examples, len(dataset))
        for i in range(num_examples):
            _, unsafe_prompt, safe_prompt = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"Unsafe Prompt: {unsafe_prompt}")
            print(f"Safe Prompt: {safe_prompt}")
        
        # Show statistics about other categories if the main category is 'sexual'
        if args.category == 'sexual':
            categories = ["self-harm", "political", "hate", "harassment"]
            print("\nStatistics for other categories:")
            for category in categories:
                try:
                    category_dataset = CoProDataset(data_path, category=category)
                    print(f"{category.capitalize()} category size: {len(category_dataset)}")
                except Exception as e:
                    print(f"Error loading {category} category: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\nCoProDataset test completed successfully!")

if __name__ == "__main__":
    main() 