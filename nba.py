import pandas as pd
from sklearn import cross_validation, linear_model
import csv
import random
import numpy


#----------------------------------------------------------
#       Init variables
#----------------------------------------------------------
teamElos = {}  
teamStats = {}
teamPlayoffs = {}
x = []
y = []
begginingYear = 2013
predictionYear = 2018
eloMoyen = 1500


#----------------------------------------------------------
#       Init Data
#----------------------------------------------------------
def InitData():

    for i in range(begginingYear, predictionYear+1):
#        teamElos[i] = {}
        teamStats[i] = {}


#----------------------------------------------------------
#       get ELO
#----------------------------------------------------------
def GetElo(team):
    
    try:
        return teamElos[team]
    except:
        return eloMoyen


#----------------------------------------------------------
#       get Stats
#----------------------------------------------------------
def GetStats(season, team):
    
    try:
        return teamStats[season][team]
    except:
        return 0
 

#----------------------------------------------------------
#       get NbPlayoffs
#----------------------------------------------------------
def GetNbPlayoffs(team):
    default = [0,0,0]

    try:
        return teamPlayoffs[team]
    except:
        return default


#----------------------------------------------------------
#       Prédiction
#----------------------------------------------------------
def Prediction(team1, team2, model):
    features = []

    #Team1
    features.append(GetElo(team1))
    features.extend(GetNbPlayoffs(team1))
    features.extend(GetStats(predictionYear-1, team1))
    
    #Team2
    features.append(GetElo(team2))
    features.extend(GetNbPlayoffs(team2))
    features.extend(GetStats(predictionYear-1, team2))
 
    return model.predict_proba([features])


#----------------------------------------------------------
#       Conctruction des données
#----------------------------------------------------------
def BuildSeasonData(seasonData, csvTeamStats, nbPlayoffsByTeam, eloScores):
    
    print("Construction des données.");

    #Parcours du dataframe des stats
    for indexS, colS in csvTeamStats.iterrows():
        teamStats[colS['Year']][colS['Team']] = []
        teamStats[colS['Year']][colS['Team']].extend((colS['FG%'], colS['3P%'], colS['2P%'], colS['FT%'], colS['ORB']/1500, colS['DRB']/3000, colS['AST']/2500, colS['STL']/800, colS['BLK']/500, colS['TOV']/1300, colS['PF']/1800, colS['PTS']/10000))
        #teamStats[colS['Year']][colS['Team']].extend((colS['FG%'], colS['3P%'], colS['2P%'], colS['FT%'], colS['ORB'], colS['DRB'], colS['STL'], colS['BLK'], colS['TOV'], colS['PF'], colS['PTS']))
        
    #Parcours du dataframe du nombre de qualifications aux playoffs
    for indexP, colP in nbPlayoffsByTeam.iterrows():
        teamPlayoffs[colP['Team']] = []
        teamPlayoffs[colP['Team']].extend((colP['NbPlayoffs'], colP['ConfChamp'], colP['NbaChamp']))

    #Parcours du dataframe des scores elos
    for indexE, colE in eloScores.iterrows():
        teamElos[colE['Team']] = colE['Elo']

        
    #Parcours du dataframe des matchs 
    for index, col in seasonData.iterrows():
        
        #Si la colonne "Winner" = H, la team gagnante est Home. Sinon le contraire
        if seasonData.iloc[index, -1] == 'H':
            winTeam = col['Home']
            loseTeam = col['Visitor']
        else:
            winTeam = col['Visitor']
            loseTeam = col['Home']

        #On rajoute les elos des équipes
        featuresTeam1 = [teamElos[winTeam]]
        featuresTeam2 = [teamElos[winTeam]]

        #On rajoute les qualifications aux playoffs des équipes
        featuresTeam1.extend(teamPlayoffs[winTeam])
        featuresTeam2.extend(teamPlayoffs[loseTeam])

        #On rajoute les stats des équipes
        featuresTeam1.extend(teamStats[col['Year']][winTeam])
        featuresTeam2.extend(teamStats[col['Year']][loseTeam])
        
        #On interchange l'ordre des équipes de façon aléatoire
        if random.random() > 0.5:
            x.append(featuresTeam1 + featuresTeam2)
            y.append(0)
        else:
            x.append(featuresTeam2 + featuresTeam1)
            y.append(1)
    
    return x, y


#----------------------------------------------------------
#       Écrire les prédiction dans un fichier csv
#----------------------------------------------------------
def WriteToCsv(finalPrediction):
    predictionLisible = []
    
    for prediction in finalPrediction:
        
        #Ordre des matchs : on met le gagnant en premier
        if prediction[2] > 0.5:
            winner = prediction[0]
            loser = prediction[1]
            proba = prediction[2]
        else:
            winner = prediction[1]
            loser = prediction[0]
            proba = 1-prediction[2]
        
        #On garde les prédiction sous forme de phrase
        predictionLisible.append(['%s vs. %s : %.2f' %(winner, loser, round(proba*100,2))])
    
    #Écriture dans le fichier csv
    with open('predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(predictionLisible)
        
    print("Fin.");
   


#----------------------------------------------------------
#       Main function
#----------------------------------------------------------
if __name__ == "__main__":
    
    #Init data
    InitData()
    
    #Lecture des fichiers csv
    print("Lecture des fichiers csv.");
    seasonData = pd.read_csv('Data/SeasonResults.csv')
    csvTeamStats = pd.read_csv('Data/TeamStats.csv')
    playoffTeams2018 = pd.read_csv('Data/PlayoffMatches2018.csv')
    nbPlayoffsByTeam = pd.read_csv('Data/PlayoffsByTeam.csv')
    eloScores = pd.read_csv('Data/TeamElos.csv')

    #Construction des données
    x,y = BuildSeasonData(seasonData, csvTeamStats, nbPlayoffsByTeam, eloScores)
    
    
#    featureScaling = preprocessing.StandardScaler()
#    featureScaling.fit(x)
#    x = featureScaling.transform(x)
    
    
    print("Taille échantillon : " + str(len(x)) + ".") 
    
    #Modèle
    model = linear_model.LogisticRegression()
    
#    weight = [1]*32
#    #Poids elo score
#    weight[0] = 2
#    weight[16] = 2
#    #Poids finales playoffs
#    weight[2] = 1.5
#    weight[18] = 1.5
#    #Poids gagnant playoffs
#    weight[3] = 2
#    weight[19] = 2
#
#    #weights = [weight]*len(x)
#    print(weight)
#    print("test")
    model.fit(x, y)
    
#    model.fit(x, y)
    
    #Test du modèle
    crossValScore = cross_validation.cross_val_score(model, numpy.array(x), numpy.array(y), cv=10).mean()
    print("Test de précision : " + str(round(crossValScore*100,2)) + "%.")
    
    #Récupération des équipes des playoffs
    playoffTeams = []
    for index, row in playoffTeams2018.iterrows():
        playoffTeams.append(row['Team'])

    #Prédiction
    print("Prédiction des matchs.");
    finalPrediction = [];
    roundPlayoffs = [];

    #Les playoffs se déroulent en 4 rounds
    for step in range(1, 5):
        print("-----------------------------------------")
        print("Round " + str(step) + " des playoffs ...")
        print("-----------------------------------------")
        
        #On parcourt les affrontements du tableau de deux en deux (2 équipes par match)
        for i in range(0,(len(playoffTeams)-1),2):
            team1 = playoffTeams[i]
            team2 = playoffTeams[i+1]
            print(team1);
            print(team2);
            print(" ");

            #Conserver le résultat de la rencontre
            prediction = Prediction(team1, team2, model)
            finalPrediction.append([team1, team2, prediction[0][0]])
            
            #Conserver le gagnant pour le prochain round
            if prediction[0][0] > 0.5:
                roundPlayoffs.append(team1);
            else:
                roundPlayoffs.append(team2);

        playoffTeams = roundPlayoffs;
        roundPlayoffs = [];


    #Écrire les résultats dans un fichier csv
    print("Écriture des résultats dans un fichier csv.");
    WriteToCsv(finalPrediction)
    