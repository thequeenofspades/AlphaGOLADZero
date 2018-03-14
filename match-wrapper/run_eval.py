import sys
import subprocess
import json

from eval_config import eval_config

def run_eval(num_matches, iters=None):
    result_json_file = "resultfile.json"

    botname = eval_config.botname
    cmd = eval_config.cmd
    player = eval_config.player

    # Set player names
    player_names = []
    for i in xrange(len(player)):
        player_names.append(botname[player[i]] + "_" + str(i))

    # [even, odd]
    winner_dict = [{'0':player_names[1],'1':player_names[0],'null':'draw'},
                   {'0':player_names[0],'1':player_names[1],'null':'draw'}]

    game_results = {}
    
    # Match index is 1-based
    for i in xrange(1, num_matches+1):
        # Read from test/wrapper-commands-template.json, modify fields when necessary
        with open("test/wrapper-commands-template.json", 'r') as in_file:
            jsonobj = json.load(in_file)

            # eval_config.players[0] is player0 (i.e. goes first) in odd-numbered matches
            jsonobj["match"]["bots"][0]["command"] = cmd[player[~(i & 1)]]
            jsonobj["match"]["bots"][1]["command"] = cmd[player[i & 1]]

            # Set board size and cells_each_player
            jsonobj["match"]["engine"]["configuration"]["fieldWidth"]["value"] \
                = eval_config.fieldWidth
            jsonobj["match"]["engine"]["configuration"]["fieldHeight"]["value"] \
                = eval_config.fieldHeight
            jsonobj["match"]["engine"]["configuration"]["initialCellsPerPlayer"]["value"] \
                = eval_config.initialCellsPerPlayer

            # Write as test/wrapper-commands.json
            out_file = open("test/wrapper-commands.json", 'w')
            out_file.write(json.dumps(jsonobj))
            out_file.close()

        # Run match, save game and log files with match index
        subprocess.call("./run_wrapper.sh")
        subprocess.call("python3 export_logs.py " + result_json_file + " game_" + str(i) + ".txt player_log_" + str(i), shell=True)
        print ("Game #" + str(i) + " completed.")

        # Extract game result, increment count
        with open(result_json_file, 'r') as in_file:
            jsonobj = json.load(in_file)
            jsonobj_details = json.loads(jsonobj["details"])
            winner = winner_dict[i & 1][jsonobj_details["winner"]]
            print ("Winner of Game #" + str(i) + ": " + winner)
            if winner not in game_results:
                game_results[winner] = 1
            else:
                game_results[winner] += 1

    game_result_filename = 'eval_results.txt'
    if iters != None:
        game_result_filename = 'eval_results_%d.txt' % int(iters)
    game_result_file = open(game_result_filename, 'w+')

    print ("Results of " + str(num_matches) + " matches:")
    game_result_file.write('Results of ' + str(num_matches) + ' matches:\n')
    print (game_results)
    game_result_file.write(str(game_results) + '\n')
    game_result_file.write('\n')
    game_result_file.close()
    
if __name__ == '__main__':
    num_matches = int(sys.argv[1])
    iters = None
    if len(sys.argv) > 2:
        iters = int(sys.argv[2])
    run_eval(num_matches, iters)
