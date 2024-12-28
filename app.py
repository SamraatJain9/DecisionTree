import os
import json  # To save the tree in a readable format

class DecisionTree:
    def __init__(self):
        self.tree = {}

    def add_root(self, question):
        self.tree['root'] = {
            'question': question,
            'yes': None,
            'no': None
        }

    def add_node(self, path, value):
        node = self.tree
        for step in path[:-1]:
            node = node[step]
        branch = path[-1]
        if "?" in value:
            node[branch] = {'question': value, 'yes': None, 'no': None}
        else:
            node[branch] = {'leaf': value}

    def print_tree(self):
        """Print the current tree structure."""
        print(self.tree)

    def visualize_tree(self, node=None, prefix=""):
        """Create a textual representation of the decision tree."""
        if node is None:
            node = self.tree['root']

        if 'question' in node:
            print(f"{prefix}Root: {node['question']}")
            if node['yes']:
                print(f"{prefix}  ├── Yes:")
                self.visualize_tree(node['yes'], prefix + "  │   ")
            if node['no']:
                print(f"{prefix}  └── No:")
                self.visualize_tree(node['no'], prefix + "      ")
        elif 'leaf' in node:
            print(f"{prefix}{node['leaf']}")

    def save_tree(self):
        """Save the current tree to a file."""
        # Find the next available filename
        i = 0
        while os.path.exists(f"tree{i}.json"):
            i += 1
        # Save the tree as JSON
        with open(f"tree{i}.json", "w") as file:
            json.dump(self.tree, file, indent=4)
        print(f"Tree saved as tree{i}.json")

# Example Usage
def create_decision_tree():
    decision_tree = DecisionTree()

    print("Welcome to the Decision Tree Builder!")

    # Add root node
    root_question = input("Enter the root node question: ")
    decision_tree.add_root(root_question)

    # Recursive tree-building process
    while True:
        print("\nOptions:")
        print("1. Add a node")
        print("2. Print the textual representation of the tree")
        print("3. Finish building the tree")
        print("4. Save the current tree")

        choice = input("Choose an option (1-4): ")

        if choice == '1':
            path = input("Enter the path to the parent node (e.g., root/yes): ").split('/')
            value = input("Enter the question (with '?') or leaf value (without '?'): ")
            decision_tree.add_node(path, value)
            print("Node added!")

        elif choice == '2':
            print("\nCurrent Tree Structure:")
            decision_tree.visualize_tree()

        elif choice == '3':
            print("Finished building the tree!")
            break

        elif choice == '4':
            decision_tree.save_tree()

        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    create_decision_tree()
