class config():
    verbose = True
    #MCTS Params
    tau = 1.
    # Number of iterations for MCTS
    mcts_itermax = 20
    # Beam width for MCTS
    beam_width = 10
    # whether to take random birth action
    do_rand_birth = True

    # Game board dimensions
    board_width = 6
    board_height = 6

    # Board initialization
    cells_each_player = 10

    # NN params
    lr = 0.01
    batch_size = 16
    n_actions = 3
    train_steps = 15000
    max_ep_length = 20
    save_freq = 5000
    print_freq = 500
    save_path = 'weights/'
    res_tower_height = 3
    
    # Outer loop params
    n_iters = 200
    buffer_size = 150
