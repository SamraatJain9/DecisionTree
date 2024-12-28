class DecisionTree:
    def __init__(self):
        self.tree = {}  # The main tree structure

    def add_root(self, question):
        self.tree['root'] = {
            'question': question,
            'yes': None,
            'no': None
        }

    def add_node(self, path, question=None, leaf_value=None):
        """Add a decision or leaf node.
        Args:
            path: List of directions ['yes', 'no', ...] to reach the parent node.
            question: Question for the new decision node (None if adding a leaf).
            leaf_value: Value for the new leaf node (None if adding a decision node).
        """
        node = self.tree
        for step in path[:-1]:
            node = node[step]
        branch = path[-1]
        if leaf_value:
            node[branch] = {'leaf': leaf_value}
        else:
            node[branch] = {'question': question, 'yes': None, 'no': None}

    def print_tree(self):
        """Print the current tree structure."""
        print(self.tree)

def create_decision_tree():
    decision_tree = DecisionTree()

    print("Welcome to the Decision Tree Builder!")

    # Add root node
    root_question = input("Enter the root node question: ")
    decision_tree.add_root(root_question)

    # Automatically add Level 1 decision nodes
    print("\nNow adding Level 1 decision nodes.")
    level1_yes_question = input("Enter question for the 'Yes' branch of the root: ")
    level1_no_question = input("Enter question for the 'No' branch of the root: ")

    decision_tree.add_node(['root', 'yes'], question=level1_yes_question)
    decision_tree.add_node(['root', 'no'], question=level1_no_question)

    # Automatically add Level 2 leaf nodes
    print("\nNow adding Level 2 leaf nodes.")
    yes_yes_leaf = input("Enter leaf value for 'Yes' branch of the 'Yes' decision node: ")
    yes_no_leaf = input("Enter leaf value for 'No' branch of the 'Yes' decision node: ")
    no_yes_leaf = input("Enter leaf value for 'Yes' branch of the 'No' decision node: ")
    no_no_leaf = input("Enter leaf value for 'No' branch of the 'No' decision node: ")

    decision_tree.add_node(['root', 'yes', 'yes'], leaf_value=yes_yes_leaf)
    decision_tree.add_node(['root', 'yes', 'no'], leaf_value=yes_no_leaf)
    decision_tree.add_node(['root', 'no', 'yes'], leaf_value=no_yes_leaf)
    decision_tree.add_node(['root', 'no', 'no'], leaf_value=no_no_leaf)

    print("\nInitial decision tree structure created!")
    decision_tree.print_tree()

    # Allow user to expand the tree further
    while True:
        print("\nOptions:")
        print("1. Add a decision node")
        print("2. Add a leaf node")
        print("3. Print the current tree")
        print("4. Finish building the tree")

        choice = input("Choose an option (1-4): ")

        if choice == '1':
            # Add a decision node
            path = input("Enter the path to the parent node (e.g., root/yes): ").split('/')
            question = input("Enter the question for this decision node: ")
            decision_tree.add_node(path, question=question)
            print("Decision node added!")

        elif choice == '2':
            # Add a leaf node
            path = input("Enter the path to the parent node (e.g., root/no): ").split('/')
            leaf_value = input("Enter the value for this leaf node: ")
            decision_tree.add_node(path, leaf_value=leaf_value)
            print("Leaf node added!")

        elif choice == '3':
            # Print the tree
            print("\nCurrent Tree Structure:")
            decision_tree.print_tree()

        elif choice == '4':
            # Exit
            print("Finished building the tree!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    create_decision_tree()
