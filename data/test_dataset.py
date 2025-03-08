import os
import sys
import argparse
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    print(f"Looking for dataset at: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist")
        sys.exit(1)
    
    print(f"Loading CoPro dataset from: {data_path}")
    print(f"Category: {args.category}")
    
    try:
        # Create dataset with the specified category
        print("Attempting to create CoProDataset instance...")
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
    
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Check that your project structure is correct and you're running from the right directory.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}")
        print("Check that the dataset files exist at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {type(e).__name__}: {e}")
        sys.exit(1)
    
    print("\nCoProDataset test completed successfully!")

if __name__ == "__main__":
    main() 