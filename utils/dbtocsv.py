## Helper functions to retrieve and organise data from the sql dataset and read into a csv for easier access

import sqlite3 as sql
import csv
import sys
import numpy as np

conn = sql.connect('database.sqlite')
c = conn.cursor()

def getPlayerData():
	"""
	Obtain relevant data from the player and player attribute tables, and write to a CSV file
	"""

	## This sql query needs to obtain the player info from the Player table, joining
	## the overall rating and the preferred foot of a player from the Player_Attributes table
	player_data = c.execute("""SELECT Player.id,Player.player_api_id,Player.player_name,ROUND(AVG(Player_Attributes.overall_rating)),Player_Attributes.preferred_foot
							FROM Player
							INNER JOIN Player_Attributes ON Player.player_api_id=Player_Attributes.player_api_id
							GROUP BY Player.id""") # there are multiple entries for each player from several fifa editions, so we use group by to just obtain one, and take the average overall rating for each player

	## Write to csv file
	with open('player-data/player_info3.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id','api_id','name','rating','foot'])
		writer.writerows(player_data)

getPlayerData()

def getMatchData():
	"""
	Obtain relevant data from the match and team tables, and write to a CSV file
	"""

	## This sql query needs to obtain the various match data from the match table,
	## and then needs to access the Team table twice, to obtain the home team's name, and then the away team's name
	c.execute("""SELECT id1, country_id, league_id, season, stage, date, match_api_id, home_player_1, away_player_1, home_name, away_name
							FROM (SELECT Match.id as id1, Match.country_id, Match.league_id,
							Match.season, Match.stage, Match.date, Match.match_api_id,
							Match.home_player_1, Match.away_player_1, Team.team_long_name as home_name
							FROM Match
							INNER JOIN Team ON Match.home_team_api_id=Team.team_api_id) AS T1
							JOIN (SELECT Match.id as id2, Team.team_long_name as away_name
							FROM Match
							INNER JOIN Team ON Match.away_team_api_id=Team.team_api_id) AS T2
							ON T1.id1=T2.id2""")

	# print(match_data)
	# sys.exit(0)

	## Need to format date field to just contain the actual date and not time
	match_data = c.fetchall() # fetchall returns a list of tuples
	match_data = np.array([list(elem) for elem in match_data]) # convert to convenient format
	match_data[:,5] = [s[:10] for s in match_data[:,5]]
	# print(match_data[:10,5])
	# sys.exit(0)

	## Write to csv file
	with open('match-data/match_info1.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id','country_id','league_id','season','stage', 'date', 'match_api_id', 'home_player_1', 'away_player_1', 'home_team', 'away_team'])
		writer.writerows(match_data)

getMatchData()
