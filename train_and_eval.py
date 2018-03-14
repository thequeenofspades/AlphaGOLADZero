import sys
import subprocess
from config import config

def main():
	for t in range(config.n_iters):
		subprocess.call('python train_loop.py', shell=True)
		subprocess.call('python ./run_eval.py %d %d' % (config.n_eval_games, t+1), shell=True, cwd='match-wrapper/')
		print "Finished one step of train + eval. Results are in match-wrapper/eval_results_%d.txt" % (t+1)

if __name__ == '__main__':
	main()