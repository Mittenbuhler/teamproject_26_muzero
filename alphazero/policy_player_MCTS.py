MCTS_POLICY_EXPLORE = 3

def Policy_Player_MCTS(mytree):  
    
    '''
    Our strategy for using AlphaZero is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to AlphaZero, is the best possible next action
    '''
    
    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()
        
    next_tree, next_action = mytree.next()
    
    # Extract the additional values from the tree nodes
    obs = next_tree.observation  # Observation after taking the action
    p_obs = mytree.observation   # Observation before taking the action
    p = mytree.visit_count_policy(temperature=1.0)  # Policy distribution at current node
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()
    
    return next_tree, next_action, obs, p, p_obs

