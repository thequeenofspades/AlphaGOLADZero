class config():
    # Number of iterations for MCTS
    mcts_itermax = 10

    # Game board dimensions
    board_width = 6
    board_height = 6

    # Board initialization
    cells_each_player = 10

    # NN params
    lr = 0.01
    batch_size = 1000
    n_actions = 3
    epochs = 20
    max_ep_length = 100
    save_freq = 100