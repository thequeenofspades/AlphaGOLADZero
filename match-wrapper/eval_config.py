class eval_config():
    # Game board
    fieldWidth = 6
    fieldHeight = 6
    initialCellsPerPlayer = 10

    # Bot list
    botname = ["random", "pass", "mcts"]
    cmd = ["python ../random_bot_main.py",
            "python ../pass_bot_main.py",
            "python ../mcts_bot_main.py"]

    # Players, using indices above
    player = [2, 0]
    