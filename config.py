class config():
    verbose = True
    # Number of iterations for MCTS
    mcts_itermax = 10
    # Beam width for MCTS
    beam_width = 10

    # Game board dimensions
    board_width = 6
    board_height = 6

    # Board initialization
    cells_each_player = 10

    # NN params
    lr = 0.01
    batch_size = 5
    n_actions = 3
    train_steps = 1000
    max_ep_length = 20
    save_freq = 10
    print_freq = 100
    save_path = 'weights/'
    res_tower_height = 19
    
    # Outer loop params
    n_iters = 100
    buffer_size = 5
