
"""
Function to read through the results of the Ray Tune experiments, and find the best performing config
"""
import os
import json
best_json = None
best_score = 0
best_dir = ''
cur_json = None
for dir in os.listdir('./ray_results/xg_tune/'):
	scores = []
	for line in open('./ray_results/xg_tune/'+dir+'/result.json',mode='r'):
		cur_json = json.loads(line)
		scores.append(cur_json['episode_reward_max'])
#       av_score = sum(scores)/len(scores)
		min_score = min(scores)
		if best_json is None or min_score > best_score:
			best_json = cur_json
			best_score = min_score
			best_dir = dir
print(best_json)
print(best_score)
print(best_dir)

with open('./ray_results/best_run_min.json', 'w') as f:
	json.dump(best_json, f)
