import sys
import json

def export_logs(in_path, game_path, playerlog_path_prefix):
    with open(in_path, 'r') as in_file:
        jsonobj = json.load(in_file)
        
        game_file = open(game_path, 'w')
        game_file.write(json.dumps(json.loads(jsonobj["game"]), indent=4))
        game_file.close()

        i = 0
        for player in jsonobj["players"]:
            log_file = open(playerlog_path_prefix + "_" + str(i) + ".txt", 'w')
            log_file.write(player["log"])
            log_file.close()
            i += 1

if __name__ == '__main__':
    export_logs(sys.argv[1], sys.argv[2], sys.argv[3])
