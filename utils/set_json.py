from utils.utils import load_json, save_json
import argparse


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Set JSON key-value pair')

    # Add arguments
    parser.add_argument('--json_file', default='./datasets.json', help='Path to the JSON file')
    parser.add_argument('--key', required=True, help='Key for the dictionary entry')
    parser.add_argument('--value', required=True, help='Value for the dictionary entry')

    # Parse the arguments
    args = parser.parse_args()

    # Load the JSON file
    data = load_json(args.json_file)

    # Set the dictionary entry with the given key-value pair
    assert args.key in data, f"Key '{args.key}' not found in the JSON file"
    data[args.key] = args.value

    # Save the updated JSON file
    save_json(data,args.json_file)

if __name__ == '__main__':
    main()