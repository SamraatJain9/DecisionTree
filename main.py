from decision_trees import train_decision_tree

def main():
    
    task_type = 'regression' 
    

    clf_best = train_decision_tree(task_type)
    
if __name__ == '__main__':
    main()
