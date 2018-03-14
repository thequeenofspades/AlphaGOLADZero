import sys
import subprocess
import json

def run_eval(num_matches, iters=None):
    result_json_file = "resultfile.json"

    winner_dict = [{'0':'random_bot','1':'mcts_bot','null':'draw'},
                   {'0':'mcts_bot','1':'random_bot','null':'draw'}]

    game_results = {}
    
    for i in xrange(1, num_matches+1):
        # Run match, save game and log files with match index
        if (i & 1):
            subprocess.call("./run_wrapper.sh")
        else:
            subprocess.call("./run_wrapper_flipped.sh")
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
    game_result_file.write(game_results)
    game_result_file.write('\n')
    game_result_file.close()
    
if __name__ == '__main__':
    num_matches = int(sys.argv[1])
    iters = None
    if len(sys.argv) > 2:
        iters = int(sys.argv[2])
    run_eval(num_matches, iters)
