"""
University of Sussex MSc Project. Building Neural Network Expected Goals Model
Warren Boult, 2018.

This file contains all the functionality used to load, preprocess and train the model.
"""

import pandas as pd
import shutil
import numpy as np
import sys
import argparse
import os
import difflib
import jellyfish as jf
import sqlite3 as sql

import tensorflow as tf
import tensorflow.contrib.learn as skflow
# import tensorflow.contrib.estimator.early_stopping as early_stopping
from google.datalab.ml import TensorBoard
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.svm import SVC
import ray.tune as tune
from sklearn.utils import shuffle

import ray
from ray.tune import grid_search, run_experiments, register_trainable, Experiment

import matplotlib.pyplot as plt
# plt.ioff() ## Turn off plotting for cloud VM

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print('Num cores: ', num_cores)

## Set data path
DATA_PATH = 'football-events'

## Make pandas display full content of entries
pd.set_option('display.max_colwidth', -1)


def buildFeatureCols():
	"""
	Extract the features to be used for model training
	"""

	print('Doing feature building')
	## Print errors to file
	# save_stderr = sys.stderr
	# fh = open("errors/errors.txt","w")
	# sys.stderr = fh

	## Feature names
	feature_names = ['minute_of_game','away_or_home','shot_place','location','foot_or_head','assist_method','goalkeeper_skill','fast_break','goal_scored']
	# feature_names = ['minute_of_game','away_or_home','shot_place','location','assist_method']

	## We only want attempt events, ignore blocks
	attempt_data = event_data[event_data['event_type']==1][event_data['shot_outcome']!=3]
	# print(attempt_data[attempt_data['player2']!=attempt_data['player2']][['text','player','player2','is_goal']])
	# print(attempt_data[attempt_data['player2']!=attempt_data['player2']][attempt_data['is_goal']==1])
	# print(attempt_data[['event_team','opponent']])
	# sys.exit(0)

	# print(pd.crosstab(attempt_data.bodypart,attempt_data.is_goal,normalize='index'))
	# sys.exit(0)

	# ## Most misses by teams
	# print(attempt_data[attempt_data['shot_outcome']==2]['event_team'].value_counts())
	# sys.exit(0)
	# print(attempt_data)

	def getGameMin():
		"""
		Minute of game feature
		"""

		minute_of_game = np.array(attempt_data['time'])
		# minute_of_game = np.floor(minute_of_game/10) # binned into 10 minute intervals
		# print(np.array(minute_of_game))
		return minute_of_game

	minute_of_game = getGameMin()
	# sys.exit(0)

	def getAwayHome():
		"""
		Away or home side feature
		"""
		away_or_home = np.array(attempt_data['side'])
		return away_or_home

	# away_or_home = getAwayHome()

	def getShotPlace():
		"""
		Shot placement feature
		Shot place not usually incorporated into standard xG models
		"""
		shot_place = np.array(attempt_data['shot_place'])
		return shot_place

	# shot_place = getShotPlace()

	def getLocation():
		"""
		Pitch location feature - NB shot location of 14 means shot is a penalty
		"""
		location = np.array(attempt_data['location'])
		return location

	# location = getLocation()

	def doPlayer(index,row):
		"""
		This function finds whether the event player was shooting with their preferred foot
		if preferred foot, value 1, if non-preferred foot, value is 2, if head, value is 3
		"""

		## Grab required attributes
		name = row['player']
		bodypart = row['bodypart']
		# print(name)

		# bodypart = attempt_data[attempt_data['player']==name]['bodypart'] # 1 is right foot, 2 is left foot, 3 is head
		if bodypart == 3: ## if head, don't need to work out preferred foot, just add that it was header

			# foot_or_head.append(3)
			return 3

		else:

			## find name match in player data, may not be exact matches due to spelling differences, so use similarity measure
			# name_match = difflib.get_close_matches(str(name),player_data['name'],1)
			match_ind = np.argmin([jf.levenshtein_distance(name.lower(), matcher.lower()) for matcher in player_data['name']])
			name_match = player_data['name'][match_ind]
			print(name_match)

			# print(name_match)
			if len(name_match) < 1: ## catch no match as a different value
				# foot_or_head.append(None)
				print('Event %d no player match on %s' % (index,name), file=sys.stderr)
				return None
				# continue

			## Grab the preferred foot of the event player
			foot = player_data[player_data['name']==name_match]['foot'].values[0]
			# print(foot.values[0])
			foot = 1 if foot == 'right' else 2 ## convert to integers for comparison with bodypart

			## Assign foot or head value
			if foot == bodypart:
				# foot_or_head.append(1)
				return 1
			else:
				# foot_or_head.append(2)
				return 2

	def getPrefFoot():
		"""
		Set up parallel processing of the preferred foot feature.
		"""

		## Prepare inputs
		inputs = attempt_data[['player','bodypart']].iterrows()

		foot_or_head = Parallel(n_jobs=num_cores)(delayed(doPlayer)(i,j) for i,j in inputs)

		return np.array(foot_or_head)

	# foot_or_head = getPrefFoot()
	# print(foot_or_head)
	# sys.exit(0)

	# foot_or_head = np.array(attempt_data['bodypart'])

	def getAssistMethod():
		"""
		Return the assist method feature
		"""

		assist_method = np.array(attempt_data['assist_method'])
		return assist_method

	# assist_method = getAssistMethod()

	# # Goalkeeper skill feature -- NB need to obtain this from fifa attributes

	def doKeeper(index,row):
		"""
		Function to get the opponent goalkeeper skill rating (goalkeeper_skill feature)
		for each event
		"""

		## Grab required attributes from event
		side = row['side'] # 1
		opponent_name = row['opponent'] # 2
		match_id = row['id_odsp'] # 3

		## Find date of match from match data
		match_date = match_data[match_data['id_odsp']==match_id]['date'].values[0] # 4
		print(match_date)
		# sys.exit(0)

		## Return matching info
		date_matches = keeper_data[keeper_data['date']==match_date] # 5
		print(date_matches)
		# sys.exit(0)

		## Do home side
		if side == 1:

			## find matches in keeper_data based on opponent team name - if side is 1, opponent is away_team, if side is 2, opponent is home_team
			# name_match = difflib.get_close_matches(str(opponent_name),date_matches['away_team'],1) # 6
			match_ind = np.argmin([jf.levenshtein_distance(opponent_name.lower(), matcher.lower()) for matcher in date_matches['away_team']])
			name_match = player_data['name'][match_ind]
			print(name_match)

			if len(name_match) < 1: ## catch no match as a different value
				# goalkeeper_skill.append(None)
				# continue
				print('Event %d no team match on %s' % (index,opponent_name), file=sys.stderr)
				return None

			## Get api id of keeper
			keeper_id = date_matches[date_matches['away_team']==name_match]['away_player_1'].values[0] # 7

		## Do away side
		else:

			## find matches in keeper_data based on opponent team name - if side is 1, opponent is away_team, if side is 2, opponent is home_team
			# name_match = difflib.get_close_matches(str(opponent_name),date_matches['home_team'],1) # 6
			match_ind = np.argmin([jf.levenshtein_distance(opponent_name.lower(), matcher.lower()) for matcher in date_matches['home_team']])
			name_match = player_data['name'][match_ind]
			print(name_match)

			if len(name_match) < 1: ## catch no match as a different value
				# goalkeeper_skill.append(None)
				# continue
				print('Event %d no team match on %s' % (index,opponent_name), file=sys.stderr)
				return None

			## Get api id of keeper
			keeper_id = date_matches[date_matches['home_team']==name_match]['home_player_1'].values[0] # 7

		## If keeper not found, add a None rating
		if np.isnan(keeper_id):
			rating = None
		## Otherwise grab rating using api id
		else:
			rating = int(player_data[player_data['api_id']==keeper_id]['rating'].values[0])
		# print(rating)
		# sys.exit(0)

		# goalkeeper_skill.append(rating)
		return rating


	def getKeeperSkill():
		"""
		Set up parallel processing of the goalkeeper_skill feature
		"""

		inputs = attempt_data[['id_odsp','opponent','side']].iterrows()

		goalkeeper_skill = Parallel(n_jobs=num_cores)(delayed(doKeeper)(i,j) for i,j in inputs)

		return np.array(goalkeeper_skill)

	# goalkeeper_skill = getKeeperSkill()
	# sys.exit(0)

	def getFastBreak():
		"""
		Specifies if the event was on a fast break, ie the attacking team getting down the field quickly, often leads to bigger chances
		"""

		fast_break = np.array(attempt_data['fast_break'])
		return fast_break

	# fast_break = getFastBreak()

	def getGoalScored():
		"""
		Was a goal scored, this is the classification label
		"""

		goal_scored = np.array(attempt_data['is_goal'])
		# print(goal_scored[goal_scored==1])
		return goal_scored

	# goal_scored = getGoalScored()

	# ## Build dict of original features for input to tensorflow
	# features = dict(zip(feature_names,[minute_of_game,away_or_home,shot_place,location,foot_or_head,assist_method,goalkeeper_skill,fast_break,goal_scored]))
	# # features = dict(zip(feature_names,[minute_of_game,away_or_home,shot_place,location,assist_method]))
	# # print(features)
	# # sys.exit(0)
	#
	# feature_df = pd.DataFrame(features)
	# print(feature_df.describe())
	# print('\n')
	# print(feature_df[feature_df['goalkeeper_skill']==None])
	#
	# feature_df.to_csv('final-data/dataset.csv', encoding='utf-8', index=False)
	#


	#**** Adding new features after further analysis of the dataset ****#
	## New features to be added are game_state (if team is winning,drawing,losing) and red_card_advantage (does team have more men on pitch than opponent)

	def doScoreAndCards(match):
		"""
		Function to calculate, for each event in a given match,
		if the event team is winning or losing the match (winning_or_losing feature),
		and if the event team has more men on the pitch than the opponent (red_advantage).
		For both new features, need to run through all match events preceding the current event
		"""

		## Set up accumulators
		cur_home_goals = 0
		cur_away_goals = 0
		cur_home_reds = 0
		cur_away_reds = 0

		winning_or_losing = []
		red_advantage = []
		event_ids = []

		## Run through all match events
		# print(len(match))
		# sys.exit(0)
		for index,event in match.iterrows():
		# for event in sortedBySortOrder(match):
			# print('Index: ' ,index)

			## Grab required attributes
			side = event['side']
			sort_order = event['sort_order']
			id = event['id_event']

			## Do home side
			if side == 1:

				## If event is a goal, add to the goal accumulators, and calculate whether the event team is winning, losing or drawing
				if event['is_goal'] == 1:
					cur_home_goals += 1

				## If event is a red card, add to the red card accumulators, and calculate the red card +/- advantage feature
				if event['event_type2'] == 14:
					cur_home_reds += 1

				## Only want attempt events in featureset
				if event['event_type'] == 1 and event['shot_outcome'] != 3:

					## Work out who is winning and who is losing
					w_l = 1 if cur_home_goals - cur_away_goals > 0 else 2 if cur_home_goals - cur_away_goals == 0 else 0
					winning_or_losing.append(w_l)

					## Work out if there is a man advantage
					r_a = 1 if cur_home_reds - cur_away_reds > 0 else 0 if cur_home_reds - cur_away_reds == 0 else 2
					red_advantage.append(r_a)

					event_ids.append(id)

			## Do away side
			else:

				## If event is a goal, add to the goal accumulators, and calculate whether the event team is winning, losing or drawing
				if event['is_goal'] == 1:
					cur_away_goals += 1

				## If event is a red card, add to the red card accumulators, and calculate the red card +/- advantage feature
				if event['event_type2'] == 14:
					cur_away_reds += 1

				## Only want attempt events in featureset
				if event['event_type'] == 1 and event['shot_outcome'] != 3:

					## Work out who is winning and who is losing
					w_l = 1 if cur_away_goals - cur_home_goals > 0 else 2 if cur_away_goals - cur_home_goals == 0 else 0
					winning_or_losing.append(w_l)

					## Work out if there is a man advantage
					r_a = 1 if cur_away_reds - cur_home_reds > 0 else 0 if cur_away_reds - cur_home_reds == 0 else 2
					red_advantage.append(r_a)

					event_ids.append(id)

		df = pd.DataFrame({'id_event':event_ids,'winning_or_losing':winning_or_losing,'red_advantage':red_advantage})

		return df
		# return winning_or_losing

	def getScoreAndCards():
		"""
		Set up parallel processing of the winning_or_losing and red_advantage features
		"""

		## Group events by match for parallel processing
		# inputs = attempt_data[['id_odsp','sort_order','event_type','is_goal','side']][:100].iterrows()
		inputs = list(event_data[['id_odsp','id_event','sort_order','event_type','event_type2','is_goal','side','shot_outcome']].groupby('id_odsp'))
		# inputs = list(attempt_data[['id_odsp','sort_order','event_type2','is_goal','side']][:2000].groupby('id_odsp'))
		print('Input length: ', len(inputs))

		# sys.exit(0)

		results = Parallel(n_jobs=num_cores,verbose=10)(delayed(doScoreAndCards)(match) for _,match in inputs)
		# print(results)

		## Results come back as list of dataframes, flatten into one dataframe
		results = pd.concat(results)
		print(results)
		# sys.exit(0)

		return results

	score_and_cards = getScoreAndCards()
	# winning_or_losing = score_and_cards[:,0]
	# red_advantage = score_and_cards[:,1]
	# # winning_or_losing, red_advantage = getScoreAndCards()
	# print(winning_or_losing)
	# sys.exit(0)

	def getIDs():
		"""
		Add the match and event ids. Used to identify events
		"""

		return attempt_data[['id_odsp','id_event']]

	ids = getIDs()

	def getSeason():
		"""
		Add the season of match. Used to split dataset
		"""

		return match_data[['id_odsp','season']]

	seasons = getSeason()
	# print(seasons)

	def getSituation():
		"""
		Grab situation feature
		"""

		return attempt_data[['id_event','situation']]

	situations = getSituation()

	def getLeague():
		"""
		Add match league for train-test-val dataset split
		"""

		return match_data[['id_odsp','league']]

	leagues = getLeague()

	## Add new features
	feature_df = pd.read_csv('final-data/datasetfinal.csv')
	# print('\n')
	# print(ids)
	# print(len(score_and_cards))
	# sys.exit(0)

	## Concat ids to the original feature set
	feature_df.reset_index(drop=True,inplace=True)
	ids.reset_index(drop=True,inplace=True)
	score_and_cards.reset_index(drop=True,inplace=True)
	# feature_df = feature_df[:67].assign(winning_or_losing=winning_or_losing, red_advantage=red_advantage)
	feature_df = pd.concat([feature_df,ids],axis=1)

	## Merge new features in
	feature_df = feature_df.merge(score_and_cards,on='id_event',how='inner')
	# print(feature_df.describe())
	# sys.exit(0)

	## Merge seasons with feature set
	feature_df = feature_df.merge(seasons,on='id_odsp',how='inner')
	# print(feature_df)
	# sys.exit(0)

	## Merge league with feature set
	feature_df = feature_df.merge(leagues,on='id_odsp',how='inner')
	# print(feature_df)
	# sys.exit(0)

	## Merge situations with feature set
	# print(feature_df['situation'])
	feature_df = feature_df.merge(situations,on='id_event',how='inner')
	print(feature_df)
	sys.exit(0)

	feature_df.to_csv('final-data/datasetfinal.csv', encoding='utf-8', index=False)
	# score_and_cards.to_csv('final-data/scorescards.csv', encoding='utf-8', index=False)

	# ## return to normal:
	# sys.stderr = save_stderr
	#
	# fh.close()

####### Exploratory analysis #######

def plotTargetVsFeatures(data,targets):
	"""
	Plot the features versus the target variable
	"""

	# fig, axs = plt.subplots(data.shape[1])
	if len(data.shape) == 1:
		fig, axs = plt.subplots(1)
		axs.scatter(data,targets)
	else:
		fig, axs = plt.subplots(data.shape[1])
		names = list(data)
		print(names)
		for i,ax in enumerate(axs):
			print(i)
			ax.scatter(data[names[i]],targets)
	plt.xlabel('Features vs targets')
	plt.show()

def histPlotFeatures(X):
	"""
	Plot histograms of the features
	"""

	# X = np.log(X)
	# if X.shape[1] == 1:
	if len(X.shape) == 1:
		fig, axs = plt.subplots(1)
		axs.hist(X)
	else:
		fig, axs = plt.subplots(X.shape[1])
		names = list(data)
		for i in range(X.shape[1]):
			X1 = X[names[i]]
			axs[i].hist(X1,20)
			axs[i].set_title(names[i])
	plt.title('Feature histograms')
	plt.show()

def exploreData(plot=False):
	"""
	Transform and analyse the data.
	Produces the final dataset for training
	"""

	print('Doing data exploration')

	## Load data
	final_data = pd.read_csv('final-data/datasetfinal.csv')
	# print(final_data.head(100))
	# print(final_data.describe())
	# print('Num null foot_or_head datapoints: ', final_data['foot_or_head'].isnull().sum())
	# print('Num null goalkeeper_skill datapoints: ', final_data['goalkeeper_skill'].isnull().sum())

	## Drop rows with NaN values
	drop_data = final_data.dropna()
	# print(drop_data.describe())
	# sys.exit(0)

	## Remove shot_place feature as this gives information about the shot outcome
	data = drop_data.drop(columns=['shot_place'])

	## Bin numerical features
	data['goalkeeper_skill'], gk_bins = pd.cut(data['goalkeeper_skill'],10,labels=False,retbins=True)

	data['minute_of_game'], min_bins = pd.cut(data['minute_of_game'],10,labels=False,retbins=True)
	# data['goalkeeper_skill'] = pd.qcut(data['goalkeeper_skill'],12)

	# print(data.minute_of_game)
	# print('\n\n')

	## Remove the location not recorded attempts, and the penalty chances
	data = data[data['location']!=19]
	data = data[data['location']!=14]
	# print(data.describe())
	# print('\n\n')

	## Separate out target and data
	targets = data['goal_scored']
	# data = data.drop(columns=['goal_scored'])

	# ## Create crosstabs of features with target
	# print(pd.crosstab(drop_data.location, drop_data.goal_scored, normalize='index'))
	# print(data['location'].value_counts())
	# print(data['goalkeeper_skill'].value_counts())
	# print(data['assist_method'].value_counts())
	# print(data['minute_of_game'].value_counts())
	# print(targets.value_counts())
	# print(pd.crosstab(drop_data.shot_place, drop_data.goal_scored, normalize='index'))
	# print(pd.crosstab(drop_data.fast_break, drop_data.goal_scored, normalize='index'))
	# print(pd.crosstab(drop_data.foot_or_head, drop_data.goal_scored, normalize='index'))
	# crosstab = pd.crosstab(data.minute_of_game, targets, normalize='index')
	# crosstab = pd.crosstab(data.away_or_home, targets, normalize='index')
	# crosstab = pd.crosstab(data.location, targets)
	# crosstab = pd.crosstab(data.goalkeeper_skill, targets)
	# crosstab = pd.crosstab(index=data.location, columns='goal')
	# print(crosstab)
	# print(data.keys())

	## Make plots
	if plot:
		crosstab.plot.bar(stacked=True)

		# plt.bar(crosstab)
		plt.ylabel('Number of samples')
		plt.xlabel('Location feature')
		# plt.xticks(rotation='horizontal')
		plt.title('Feature bar plot')
		plt.show()

	# print(pd.crosstab(data.goalkeeper_skill,targets,normalize='index'))
	# plotTargetVsFeatures(data['location'],targets)
	# histPlotFeatures(data)
	# sys.exit(0)

	return data


##### Tensorflow model building training evaluating ######

def loadModel(model_dir,feature_cols,train_input_fn):
	"""
	Load saved model for evaluation
	"""

	new_estimator = tf.estimator.DNNClassifier(
						hidden_units=[100, 50],
						feature_columns=feature_cols.values(),
						dropout=0.3,
						optimizer=tf.train.AdamOptimizer(learning_rate=0.0005),
						model_dir=model_dir)
	new_estimator.train(input_fn=train_input_fn,steps=1) # Hacky method, train for one step to properly load back in the model

	return new_estimator

def doModel():
	"""
	Functions to set up the model, train and evaluate
	"""

	print('Build, train, eval, predict or simulate with model')
	# final_df = pd.read_csv('final-data/datasetfinal.csv')
	# print(final_df['goalkeeper_skill'])
	# sys.exit(0)

	## Tensorflow DNN needs integer inputs, so convert float64 columns to int64
	final_df['foot_or_head'] = final_df['foot_or_head'].astype(int)
	final_df['location'] = final_df['location'].astype(int)
	final_df['situation'] = final_df['location'].astype(int)
	# print(final_df.dtypes)

	# print(final_df.describe())
	# print(final_df['location'].head(20))
	# sys.exit(0)

	## get feature names, ignoring the metadata columns
	feature_names = [name for name in final_df.keys() if name not in ['goal_scored','id_odsp','id_event','season','league']]

	# print(final_df[feature_names].drop_duplicates().shape[0])
	# sys.exit(0)

	# mask = final_df['season']<2016
	# train_df = final_df[mask]
	# eval_df = final_df[~mask]
	mask1 = final_df['season']<2016 ## Split into train and test splits, test set is only 15/16 season
	train_df = final_df[mask1]
	eval_df = final_df[~mask1]
	mask2 = eval_df['league'] == 'E0' ## Further split test set into eval and validation sets, validation set is only EPL games
	val_df = eval_df[mask2]
	eval_df = eval_df[~mask2]
	# print('Train: ', train_df.describe())
	# print('Eval: ', eval_df.describe())
	# print('Val: ', val_df.describe())
	# print('\n------------------------\n')
	# sys.exit(0)

	# train_df.to_csv('final-data/traindata.csv', encoding='utf-8', index=False)
	# val_df.to_csv('final-data/prem16data.csv', encoding='utf-8', index=False)
	# sys.exit(0)

	def buildFeatureCols():
		"""
		Create the feature columns for specifying tensor inputs to the DNNClassifier
		"""

		## Build tf feature columns for each of the input features
		feature_cols = {
			name: tf.feature_column.indicator_column(
					tf.feature_column.categorical_column_with_vocabulary_list(
						key=name,
						vocabulary_list=final_df[name].unique()))
				for name in feature_names if name not in ['foot_or_head','assist_method','situation','location','winning_or_losing','away_or_home']
		}

		feature_cols['foot_or_head'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='foot_or_head',
										vocabulary_list=final_df['foot_or_head'].unique())

		feature_cols['assist_method'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='assist_method',
										vocabulary_list=final_df['assist_method'].unique())

		feature_cols['situation'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='situation',
										vocabulary_list=final_df['situation'].unique())

		feature_cols['location'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='location',
										vocabulary_list=final_df['location'].unique())

		feature_cols['winning_or_losing'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='winning_or_losing',
										vocabulary_list=final_df['winning_or_losing'].unique())

		feature_cols['away_or_home'] = tf.feature_column.categorical_column_with_vocabulary_list(
										key='away_or_home',
										vocabulary_list=final_df['away_or_home'].unique())

		## Add crossed columns, maybe a new feature crossing the body part of the chance with the type of assist will be valuable,
		## likewise maybe body part crossed with the play situation will be valuable.
		## Location cross assist_method may give a link between certain types of play leading
		## to better goal chances. Eg a through ball to a central box location usually indicates a dangerous chance
		feature_cols['foot_x_assist'] = tf.feature_column.indicator_column(
										tf.feature_column.crossed_column(
											[feature_cols['foot_or_head'],feature_cols['assist_method']],hash_bucket_size=int(9)))

		feature_cols['foot_x_situ'] = tf.feature_column.indicator_column(
										tf.feature_column.crossed_column(
											[feature_cols['foot_or_head'],feature_cols['situation']],hash_bucket_size=int(9)))

		feature_cols['loc_x_assist'] = tf.feature_column.indicator_column(
										tf.feature_column.crossed_column(
											[feature_cols['location'],feature_cols['assist_method']],hash_bucket_size=int(9)))

		feature_cols['winning_x_side'] = tf.feature_column.indicator_column(
										tf.feature_column.crossed_column(
											[feature_cols['winning_or_losing'],feature_cols['away_or_home']],hash_bucket_size=int(9)))

		feature_cols['foot_or_head'] = tf.feature_column.indicator_column(feature_cols['foot_or_head'])
		feature_cols['assist_method'] = tf.feature_column.indicator_column(feature_cols['assist_method'])
		feature_cols['situation'] = tf.feature_column.indicator_column(feature_cols['situation'])
		feature_cols['location'] = tf.feature_column.indicator_column(feature_cols['location'])
		feature_cols['winning_or_losing'] = tf.feature_column.indicator_column(feature_cols['winning_or_losing'])
		feature_cols['away_or_home'] = tf.feature_column.indicator_column(feature_cols['away_or_home'])

		return feature_cols

	feature_cols = buildFeatureCols()

	# file_writer = tf.summary.FileWriter('./xg-trained', sess.graph)

	def oversampleData(df,min_size,maj_size):
		"""
		Oversample minority class
		"""

		temp_x = df[feature_names]
		temp_y = df['goal_scored']
		# print('Before OS: ', len(temp_x[temp_y==1]))

		## Randomly oversample to new sample sizes
		input_x, input_y = RandomOverSampler(ratio={0: min_size, 1: maj_size},random_state=42).fit_sample(temp_x,temp_y)
		input_x = pd.DataFrame(input_x)
		input_y = pd.DataFrame({'goal_scored': input_y})
		input_x.columns = temp_x.columns

		input_x.astype(int)
		input_y.astype(int)
		# print('After OS: ', len(input_x[input_y==1]))
		# print('After OS: ', input_x[:20])
		return input_x, input_y

	input_x, input_y = oversampleData(train_df, len(train_df[train_df['goal_scored']==0]), len(train_df[train_df['goal_scored']==1])*3)
	# print(input_y.keys())
	# sys.exit(0)

	## Set some params for neural net
	NUM_EPOCHS = 1000
	BATCH_SIZE = 1
	MODEL_DIR = './xg-trained'
	MAX_STEPS = int(np.ceil(len(input_x)/BATCH_SIZE))*NUM_EPOCHS
	# MAX_STEPS = 5000

	sess = tf.Session()

	## Specify tf input functions
	train_input_fn = tf.estimator.inputs.pandas_input_fn(
							x=input_x,
							y=input_y['goal_scored'],
							num_epochs=NUM_EPOCHS,
							batch_size=BATCH_SIZE,
							shuffle=True)

	eval_input_fn = tf.estimator.inputs.pandas_input_fn(
							x=eval_df[feature_names],
							y=eval_df['goal_scored'],
							num_epochs=1,
							batch_size=len(eval_df),
							shuffle=False)

	validation_input_fn = tf.estimator.inputs.pandas_input_fn(
							x=val_df[feature_names],
							y=val_df['goal_scored'],
							num_epochs=1,
							batch_size=len(val_df),
							shuffle=False)

	def train_eval():
		"""
		Specify and build the DNN classifier model in tensorflow
		"""

		print('Setting up and training model')

		## Create classifier with specifications
		estimator = tf.estimator.DNNClassifier(
						model_dir=MODEL_DIR,
						feature_columns=feature_cols.values(),
						hidden_units=[100,50],
						dropout=0.3,
						activation_fn=tf.nn.relu,
						optimizer=tf.train.AdamOptimizer(
									learning_rate=0.001))

		## Add extra metrics
		def my_recall(labels, predictions):
			pred_values = tf.cast(predictions['class_ids'],tf.int64)
			return {'recall': tf.metrics.recall(labels,pred_values)}

		# estimator.eval_dir('./xg-trained/eval')
		os.makedirs(estimator.eval_dir())
		estimator = tf.contrib.estimator.add_metrics(estimator,my_recall)

		# # Stop early if the auc_precision_recall is good enough
		# early_stopping = tf.contrib.estimator.stop_if_higher_hook(
		# 					estimator,
		# 					metric_name='auc_precision_recall',
		# 					threshold=0.75,
		# 					min_steps=100)
		# 					# eval_dir='./xg-trained/eval')


		## Set up training and evaluation specifications
		train_spec=tf.estimator.TrainSpec(
									input_fn = train_input_fn,
									max_steps = MAX_STEPS)
									# hooks = [early_stopping])
		eval_spec=tf.estimator.EvalSpec(
									input_fn = eval_input_fn,
									steps = 500,
									start_delay_secs = 10,
									throttle_secs = 10)
									# hooks = [early_stopping])

		## Train model
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

		metrics = estimator.evaluate(input_fn=validation_input_fn)
		print('*' * 50)
		print('Metrics: ', metrics)

		return estimator

	if args.train:
		# Train and evaluate
		shutil.rmtree(MODEL_DIR, ignore_errors = True) # restart from scratch
		tf.logging.set_verbosity(tf.logging.INFO)
		estimator = train_eval()
		# sys.exit(0)

	# Specify and build the DNN classifier model in tensorflow, for hyperparameter tuning
	def train_eval_tune(config=None, reporter=None):
		"""
		Set up classifier for hyperparam tuning
		"""

		## Config gives the parameter values to use
		activation_fn = getattr(tf.nn, config['activation'])
		optimizer = getattr(tf.train, config['optimizer'])
		estimator = tf.estimator.DNNClassifier(
						model_dir=MODEL_DIR,
						feature_columns=feature_cols.values(),
						hidden_units=config['hidden_units'],
						# hidden_units=[20],
						dropout=config['dropout'],
						# dropout=0.1,
						activation_fn=activation_fn,
						optimizer=optimizer(
						# optimizer=tf.train.AdamOptimizer(
									learning_rate=config['learning_rate']))
									# learning_rate=0.0001))

		## Add extra metrics
		def my_recall(labels, predictions):
			pred_values = tf.cast(predictions['class_ids'],tf.int64)
			return {'recall': tf.metrics.recall(labels,pred_values)}

		os.makedirs(estimator.eval_dir())
		estimator = tf.contrib.estimator.add_metrics(estimator,my_recall)

		## Stop early if the PR-AUC stops improving
		early_stopping2 = tf.contrib.estimator.stop_if_no_increase_hook(
							estimator,
							metric_name='auc_precision_recall',
							max_steps_without_increase=5000,
							run_every_secs=20,
							min_steps=100000)
							# eval_dir='./xg-trained/eval')

		## Stop early if the recall gets too high
		early_stopping3 = tf.contrib.estimator.stop_if_higher_hook(
							estimator,
							metric_name='recall',
							threshold=0.6,
							run_every_secs=20,
							min_steps=100000)
							# eval_dir='./xg-trained/eval')

		## Stop early if the loss crosses threshold
		early_stopping4 = tf.contrib.estimator.stop_if_lower_hook(
							estimator,
							metric_name='loss',
							threshold=1000,
							#max_steps_without_decrease=500,
							run_every_secs=20,
							min_steps=100000)

		## Stop early if average loss crosses threshold
		early_stopping5 = tf.contrib.estimator.stop_if_lower_hook(
							estimator,
							metric_name='average_loss',
							threshold=0.2,
							run_every_secs=20,
							min_steps=100000)

		## Set up train and eval specifications
		train_spec=tf.estimator.TrainSpec(
		input_fn = train_input_fn,
		max_steps = MAX_STEPS,
		hooks = [early_stopping2,early_stopping1,early_stopping4,early_stopping5])

		eval_spec=tf.estimator.EvalSpec(
					input_fn = eval_input_fn,
					steps = None,
					start_delay_secs = 1,
					throttle_secs = 30)

		## Train model
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

		metrics = estimator.evaluate(input_fn=validation_input_fn)
		print('*' * 50)
		print('Metrics: ', metrics)

		## Use P-R auc score to evaluate hyperparameter experiments, better than ROC AUC due to class imbalance
		if reporter:
			reporter(timesteps_total=MAX_STEPS,episode_reward_max=metrics['auc_precision_recall'])

		return estimator

	labels = val_df['goal_scored'].values
	event_ids = val_df['id_event'].values
	match_ids = val_df['id_odsp'].values

	def classifySVM(train_df):
		"""
		Using SVM to make quick comparison with the DNN, sanity check
		"""

		print('Doing SVM classification')

		## SVM classifier comparison
		svm = SVC(probability=True)
		train_df = shuffle(train_df[:10000], random_state=0)
		svm = svm.fit(train_df[feature_names],train_df['goal_scored'])
		score = svm.score(val_df[feature_names][:100],val_df['goal_scored'][:100])
		predictions = svm.predict_proba(val_df[feature_names][:50])
		print('SVM score: ', score)
		for index,p in enumerate(list(predictions)[:25]):
			print('Prediction xg: ', p)
			# print('Predicted xg: ', p['probabilities'])
			print('Goal scored or not: ', labels[index])
			print(event_data[event_data['id_event']==event_ids[index]][['event_team','time']])
			print(event_data[event_data['id_event']==event_ids[index]]['text'])
			print(match_data[match_data['id_odsp']==match_ids[index]][['link_odsp','season']])

	if args.svm_classify:
		classifySVM(train_df)
		# sys.exit(0)

	def predict(estimator=None,model_dir=None):
		"""
		Make predictions on the evaluation dataset, 2015/16 EPL data
		"""

		print('Making predictions')

		## Write predictions to file
		tempstd = sys.stdout
		sys.stdout = open('outputs.txt','w')
		if estimator is None:

			if model_dir is None:
				print('No model_dir specified')
				return

			estimator = loadModel(model_dir,feature_cols,train_input_fn)

		# metrics = estimator.evaluate(input_fn=validation_input_fn)
		predictions = estimator.predict(input_fn=validation_input_fn)
		# print(predictions)

		for index,p in enumerate(list(predictions)):
			print('Prediction attributes: ', p)
			print('Predicted xg: ', p['probabilities'])
			print('Goal scored or not: ', labels[index])
			print(event_data[event_data['id_event']==event_ids[index]][['event_team','time']])
			print(event_data[event_data['id_event']==event_ids[index]]['text'])
			print(match_data[match_data['id_odsp']==match_ids[index]][['link_odsp','season']])

		sys.stdout = tempstd

	if args.predict:

		if args.predict == 'best':
			predict(model_dir='model_100_50/xg_trained')
		else:
			predict(model_dir=MODEL_DIR)
		# sys.exit(0)

	def paramTune():
		"""
		Set up and perform the hyperparameter tuning experiments.
		Uses the Ray Tune package.
		"""

		print('Setting up hyperparam experiments and running')

		## Set up Ray parameter tuning experiments
		RAY_DIR = './ray_results'
		register_trainable('train_xg', train_eval_tune)
		xg_spec = Experiment(
			name='xg_tune',
			run='train_xg',
			stop={
				'episode_reward_max': 0.6,
			},
			config={
				'activation': grid_search(['relu', 'leaky_relu']),
				'hidden_units': grid_search([[20],[64],[32,12],[100,50],[128,64,32]]),
				'dropout': grid_search([0.1,0.25,0.4]),
				'optimizer': grid_search([
								'AdamOptimizer',
								'GradientDescentOptimizer',
								'AdagradOptimizer',
								'RMSPropOptimizer'
							]),
				'learning_rate': grid_search([0.001,1e-4,1e-6,1e-8])
			},
			trial_resources={'cpu':1,'gpu':1},
			repeat=3,
			local_dir=RAY_DIR)

		## Delete old experiments, initialise ray, run trials
		shutil.rmtree(RAY_DIR, ignore_errors = True) # restart from scratch
		ray.init(num_cpus=num_cores)
		experiments = run_experiments(experiments=xg_spec)

	if args.param_tune:
		paramTune()

	if args.tensorboard:
		pid = TensorBoard().start(args.tensorboard)
		# TensorBoard().stop(pid)

	def evalBestModel(model_dir):
		"""
		Read in a saved model, build a DNNClassifier out of it, and then evaluate
		"""

		print('Evaluate model with metrics')

		new_estimator = loadModel(model_dir,feature_cols,train_input_fn)
		# ## Load back in trained model using same model_dir
		# new_estimator = tf.estimator.DNNClassifier(
		# 					hidden_units=[100, 50],
		# 					feature_columns=feature_cols.values(),
		# 					dropout=0.3,
		# 					optimizer=tf.train.AdamOptimizer(learning_rate=0.0005),
		# 					model_dir=model_dir)
		# new_estimator.train(input_fn=train_input_fn,steps=1) # Hacky method, train for one step to properly load back in the model

		## Evaluate model
		# metrics = new_estimator.evaluate(input_fn=validation_input_fn)
		metrics = new_estimator.evaluate(input_fn=eval_input_fn)
		print('*' * 50)
		print('Metrics: ', metrics)
		# predict(new_estimator)

	if args.eval:

		if args.eval == 'best':
			evalBestModel('model_100_50/xg-trained')
		else:
			evalBestModel(MODEL_DIR)
		# sys.exit()

	def simulateSeason(model_dir):
		"""
		This function uses the xG scores of the estimator to predict match scores in the 2016/17 EPL season,
		and then build a simulated league table for comparison with the actual league table

		One game from the 15/16 EPL season is missing data, Sunderland vs Southampton (match id b17Jzm9N/),
		so the actual score of this game is used for the simulation.
		"""

		print('Simulate a season with xG model')

		## Helper function to update the league table with results
		def updateTable(table,ht, at, hg, ag):
			"""
			Helper function to update the simulated league table with results
			"""

			## Get row index for home/away teams
			home_ind = table.set_index('Team').index.get_loc(ht)
			away_ind = table.set_index('Team').index.get_loc(at)

			## Cast to int so that xG values are converted to ints
			hg = int(hg)
			ag = int(ag)

			## Work out who won
			hr = 'win' if hg - ag > 0 else 'draw' if hg - ag == 0 else 'lose'
			# away_result = 'win' if home_result == 'lose' else 'draw' if home_result == 'draw' else 'lose'

			## Update matches played
			table.ix[home_ind,'MP'] += 1
			table.ix[away_ind,'MP'] += 1

			## Update goals columns
			table.ix[home_ind,'F'] += hg
			table.ix[home_ind,'A'] += ag
			table.ix[away_ind,'F'] += ag
			table.ix[away_ind,'A'] += hg
			table.ix[home_ind,'GD'] += (hg-ag)
			table.ix[away_ind,'GD'] += (ag-hg)

			## Update win, draw and loss columns
			## Then update points columns
			if hr == 'win':
				table.ix[home_ind,'W'] += 1
				table.ix[away_ind,'L'] += 1
				table.ix[home_ind,'P'] += 3

			elif hr == 'draw':
				table.ix[home_ind,'D'] += 1
				table.ix[away_ind,'D'] += 1
				table.ix[home_ind,'P'] += 1
				table.ix[away_ind,'P'] += 1

			elif hr == 'lose':
				table.ix[away_ind,'W'] += 1
				table.ix[home_ind,'L'] += 1
				table.ix[away_ind,'P'] += 3

			print(table.sort_values(by=['P','GD'],ascending=False))
			return table

		# feature_cols = buildFeatureCols()
		xg_estimator = loadModel(model_dir,feature_cols,train_input_fn)

		## Read in required csv files
		eval_data = pd.read_csv('final-data/prem16data.csv')

		event_data = pd.read_csv('football-events/events.csv')
		event_data = event_data[event_data['event_type']==1][event_data['shot_outcome']!=3]
		# data2016 = event_data[event_data['event_type']==1][event_data['shot_outcome']!=3]

		match_data = pd.read_csv('football-events/ginf.csv')
		event_data2016 = event_data.merge(match_data[['league','season','id_odsp']],on='id_odsp',how='inner')
		event_data2016 = event_data2016[event_data2016['season']==2016][event_data2016['league']=='E0']
		# print(event_data2016)

		## Counting total number of matches that have data
		match_data2016 = match_data[match_data['season']==2016][match_data['league']=='E0']
		match_list = match_data2016['id_odsp'].unique()
		teamlist = match_data2016['ht'].unique()

		# big_list = event_data2016['id_odsp'].unique()
		small_list = eval_data['id_odsp'].unique()

		# [print(match_data[['ht','at']][match_data['id_odsp']==id]) for id in match_list if id not in small_list]

		## Construct initial league table, constuct baseline table for comparison, where all attempts are given 0.133xG
		league_table = pd.DataFrame({'Team':teamlist,'MP':np.zeros(20,dtype=np.int8),'W':np.zeros(20,dtype=np.int8),'D':np.zeros(20,dtype=np.int8),'L':np.zeros(20,dtype=np.int8),'F':np.zeros(20,dtype=np.int8),'A':np.zeros(20,dtype=np.int8),'GD':np.zeros(20,dtype=np.int8),'P':np.zeros(20,dtype=np.int8)})
		baseline_table = pd.DataFrame({'Team':teamlist,'MP':np.zeros(20,dtype=np.int8),'W':np.zeros(20,dtype=np.int8),'D':np.zeros(20,dtype=np.int8),'L':np.zeros(20,dtype=np.int8),'F':np.zeros(20,dtype=np.int8),'A':np.zeros(20,dtype=np.int8),'GD':np.zeros(20,dtype=np.int8),'P':np.zeros(20,dtype=np.int8)})

		tally = 0
		for match_id in match_list:
			tally += 1
			print('Simming match {} of 380'.format(tally))

			## Get team names
			home_team = match_data2016[match_data2016['id_odsp']==match_id]['ht'].values[0]
			away_team = match_data2016[match_data2016['id_odsp']==match_id]['at'].values[0]

			## One match had no events available, so catch that
			if match_id not in small_list:
				home_goals = match_data2016[match_data2016['id_odsp']==match_id]['fthg']
				away_goals = match_data2016[match_data2016['id_odsp']==match_id]['ftag']

				league_table = updateTable(league_table, home_team, away_team, home_goals, away_goals)
				continue

			else:
				## Get samples, and their event ids from dataset for the match
				eval_match = eval_data[eval_data['id_odsp']==match_id]
				eval_id_list = eval_data[eval_data['id_odsp']==match_id]['id_event'].unique()
				# print(eval_match)
				# sys.exit(0)

				home_xg = 0
				away_xg = 0
				base_home_xg = 0
				base_away_xg = 0

				# print(event_data2016[event_data2016['id_odsp']==match_id].values)
				# sys.exit(0)

				## Iterate over all the attempt events in the unprocessed dataset
				for index,event in event_data2016[event_data2016['id_odsp']==match_id].iterrows():

					xg = 0
					# print(event)

					## If event is penalty, simply add 0.76xG
					if event['location'] == 14:
						xg = 0.76

					## If event has info, predict xg using classifier
					elif event['id_event'] in eval_id_list:

						# print(eval_match[eval_match['id_event']==match_id][feature_names])
						# print(len(eval_match[eval_match['id_event']==match_id][feature_names]))
						## Create input function for event, obtain output probability as xG
						input_fn = tf.estimator.inputs.pandas_input_fn(
												x=eval_match[eval_match['id_event']==event['id_event']][feature_names],
												y=None,
												num_epochs=1,
												shuffle=False,
												batch_size=1)

						xg = list(xg_estimator.predict(input_fn=input_fn))[0]['logistic']
						# xg = list(xg_estimator.predict(input_fn=input_fn))
						print(xg)
						# sys.exit(0)

					## If event is not found in evaluation set, then simply give baseline score of 0.133xG
					else:
						xg = 0.133

					## Sum up home and away xg
					if event['side'] == 1:
						home_xg += xg
						base_home_xg += 0.133
					else:
						away_xg += xg
						base_away_xg += 0.133

				## Update tables
				league_table = updateTable(league_table, home_team, away_team, home_xg, away_xg)
				baseline_table = updateTable(baseline_table, home_team, away_team, base_home_xg, base_away_xg)

		## Sort final league tables by points and goal difference
		final_table = league_table.sort_values(by=['P','GD'],ascending=False)
		final_baseline_table = baseline_table.sort_values(by=['P','GD'],ascending=False)
		print('\n\nFinal xG table: ', final_table)
		print('\n\nFinal baseline table: ', final_baseline_table)

		final_table.to_csv('eval-table/final_table.csv', encoding='utf-8', index=False)
		final_baseline_table.to_csv('eval-table/baseline_table.csv', encoding='utf-8', index=False)

		# print(match_data['ht'][match_data['season']==2016][match_data['league']=='E0'].value_counts())
		# print(match_data['at'][match_data['season']==2016][match_data['league']=='E0'].value_counts())

	if args.simulate:

		if args.simulate == 'best':
			simulateSeason('model_100_50/xg-trained')
		else:
			simulateSeason(MODEL_DIR)
		sys.exit(0)


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Arguments specify which step to perform')
	parser.add_argument('--build_features', help='build features for model and save to file', default=False, action='store_true')
	parser.add_argument('--train', help='train model save to xg-trained', default=False, action='store_true')
	parser.add_argument('--svm_classify', help='classify data with svm for comparison', default=False, action='store_true')
	parser.add_argument('--eval', help='eval trained model loaded from specificed model directory - if "best", load best model', default=False, action='store')
	parser.add_argument('--predict', help='predict on trained model loaded from specificed model directory - if "best", load best model', default=False, action='store')
	parser.add_argument('--param_tune', help='Run hyperparameter tuning experiments', default=False, action='store_true')
	parser.add_argument('--simulate', help='Simulate Premier League season with xG from specified model - if "best", load best model', default=False, action='store')
	parser.add_argument('--tensorboard', help='Start tensorboard from specified model directory', default=False, action='store')
	args = parser.parse_args()

	## Data about game events
	event_data = pd.read_csv(DATA_PATH+'/events.csv')
	# event_data = event_data[event_data['situation']==1]

	## Data about the matches, including match ID, season
	match_data = pd.read_csv(DATA_PATH+'/ginf.csv')

	## Data about players
	player_data = pd.read_csv('player-data/player_info3.csv')

	## Data linking matches to the playing goalkeepers
	keeper_data = pd.read_csv('match-data/match_info1.csv')

	# print(event_data[event_data['event_type']==1][event_data['shot_outcome']!=3])
	# print(event_data['fast_break'].sum())
	# print(event_data[event_data['is_goal']==1][event_data['fast_break']==1])

	## Just want to build features, specify --build_features arg
	if args.build_features:
		buildFeatureCols()
		sys.exit(0)

	final_df = exploreData(plot=False)
	doModel()
