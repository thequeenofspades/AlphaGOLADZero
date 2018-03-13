import sys
import subprocess
import json

def run_eval(num_matches):
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

    print ("Results of " + str(num_matches) + " matches:")
    print (game_results)
    
if __name__ == '__main__':
    run_eval(int(sys.argv[1]))
