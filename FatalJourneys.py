'''Pranav Amarnath - Fatal Journeys Project'''
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#from matplotlib import style # for style (line #12)
import pandas as pd
#from pandas.plotting import register_matplotlib_converters  # -> for scatter plots
import os
#import seaborn as sns
from wordcloud import WordCloud

#style.use("seaborn")
'''
To fill the window full screen automatically - figsize=(20, 10)
There are 6229 lines in MissingMigrantsWebsite.csv
line 4980 -> April 18, 2015 , 750 , 272 , 1022 , 28 , Drowning, 120mi. south of Lampedusa, 70mi. north of Libya Central Mediterranean
'''
pointsToDisplay = 6228
#pointSize = 10
#register_matplotlib_converters() # -> for scatter plots

'''NEED TO SPECIFY CORRECT PATH WHJEN RUNNING CODE!!'''
download_folder = os.path.expanduser("~")+"\\Desktop\\Data Science Projects\\Projects\\FatalJourneys\\IOM_FatalJourneys_Dataset.csv"
df = pd.read_csv(download_folder)
#df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d-%b-%y')
#above^ -> for IOM_FatalJourneys_Dataset.csv
df['Cause of Death'].fillna('Unknown', inplace=True)
df['Region of Incident'].fillna('Unknown', inplace=True)
df['Migration Route'].fillna('Unknown', inplace=True)

def Dead():
    sixth = df.loc[df['Reported Year'] == 2019, 'Number Dead'].sum()
    fifth = df.loc[df['Reported Year'] == 2018, 'Number Dead'].sum()
    fourth = df.loc[df['Reported Year'] == 2017, 'Number Dead'].sum()
    third = df.loc[df['Reported Year'] == 2016, 'Number Dead'].sum()
    second = df.loc[df['Reported Year'] == 2015, 'Number Dead'].sum()
    first = df.loc[df['Reported Year'] == 2014, 'Number Dead'].sum()

    print('Number of Dead')
    print('2014: ', end="")
    print(first)
    print('2015: ', end="")
    print(second)
    print('2016: ', end="")
    print(third)
    print('2017: ', end="")
    print(fourth)
    print('2018: ', end="")
    print(fifth)
    print('2019: ', end="")
    print(sixth)

    data = {'Reported Year': ['2014', '2015', '2016', '2017', '2018', '2019'], 'Number Dead': [first, second, third, fourth, fifth, sixth]}
    dfDeadByYear = pd.DataFrame.from_dict(data)

    x = dfDeadByYear.loc[0:pointsToDisplay, 'Reported Year']
    #x = df2.loc[df2.groupby("Reported Date")["Number Dead"]]
    y = dfDeadByYear.loc[0:pointsToDisplay, 'Number Dead']
    #y = df2.loc[df2.groupby("Reported Date")["Number Dead"]]
    figure = plt.figure('Dead')
    figure.suptitle('The Numbers of Dead Migrants Throughout the World (2014-2019)')
    #plot = plt.plot(x,y)
    plot = plt.bar(x, y, align='center', alpha=1)
    #plot = plt.scatter(x, y, s=pointSize)
    #plt.legend(plot[:6], ['Number Dead'])

def Survivors():
    sixth = df.loc[df['Reported Year'] == 2019, 'Number of Survivors'].sum()
    fifth = df.loc[df['Reported Year'] == 2018, 'Number of Survivors'].sum()
    fourth = df.loc[df['Reported Year'] == 2017, 'Number of Survivors'].sum()
    third = df.loc[df['Reported Year'] == 2016, 'Number of Survivors'].sum()
    second = df.loc[df['Reported Year'] == 2015, 'Number of Survivors'].sum()
    first = df.loc[df['Reported Year'] == 2014, 'Number of Survivors'].sum()

    print('\nNumber of Survivors')
    print('2014: ', end="")
    print(first)
    print('2015: ', end="")
    print(second)
    print('2016: ', end="")
    print(third)
    print('2017: ', end="")
    print(fourth)
    print('2018: ', end="")
    print(fifth)
    print('2019: ', end="")
    print(sixth)

    data = {'Reported Year': ['2014', '2015', '2016', '2017', '2018', '2019'], 'Number of Survivors': [first, second, third, fourth, fifth, sixth]}
    dfSurvivorsByYear = pd.DataFrame.from_dict(data)

    x = dfSurvivorsByYear.loc[0:pointsToDisplay, 'Reported Year']
    y = dfSurvivorsByYear.loc[0:pointsToDisplay, 'Number of Survivors']
    figure = plt.figure('Survived')
    figure.suptitle('The Numbers of Surviving Migrants Throughout the World (2014-2019)')
    #plot = plt.plot(x,y)
    plot = plt.bar(x, y, align='center', alpha=1)
    #plot = plt.scatter(x, y, s=pointSize)
    plt.xlabel('Reported Years')
    plt.ylabel('Number of Survivors')
    #plt.legend(plot[:6], ['Number Survived'])

def Missing():
    sixth = df.loc[df['Reported Year'] == 2019, 'Minimum Estimated Number of Missing'].sum()
    fifth = df.loc[df['Reported Year'] == 2018, 'Minimum Estimated Number of Missing'].sum()
    fourth = df.loc[df['Reported Year'] == 2017, 'Minimum Estimated Number of Missing'].sum()
    third = df.loc[df['Reported Year'] == 2016, 'Minimum Estimated Number of Missing'].sum()
    second = df.loc[df['Reported Year'] == 2015, 'Minimum Estimated Number of Missing'].sum()
    first = df.loc[df['Reported Year'] == 2014, 'Minimum Estimated Number of Missing'].sum()

    print('\nNumber of Missing')
    print('2014: ', end="")
    print(first)
    print('2015: ', end="")
    print(second)
    print('2016: ', end="")
    print(third)
    print('2017: ', end="")
    print(fourth)
    print('2018: ', end="")
    print(fifth)
    print('2019: ', end="")
    print(sixth)

    data = {'Reported Year': ['2014', '2015', '2016', '2017', '2018', '2019'], 'Minimum Estimated Number of Missing': [first, second, third, fourth, fifth, sixth]}
    dfMissingByYear = pd.DataFrame.from_dict(data)

    x = dfMissingByYear.loc[0:pointsToDisplay, 'Reported Year']
    y = dfMissingByYear.loc[0:pointsToDisplay, 'Minimum Estimated Number of Missing']
    figure = plt.figure('Missing')
    figure.suptitle('The Numbers of Missing Migrants Throughout the World (2014-2019)')
    #plot = plt.plot(x,y)
    plot = plt.bar(x, y, align='center', alpha=1)
    #plot = plt.scatter(x, y, s=pointSize)
    plt.xlabel('Reported Years')
    plt.ylabel('Number of Missing')
    #plt.legend(plot[:6], ['Number of Missing'])

def DeadAndMissing():
    sixth = df.loc[df['Reported Year'] == 2019, 'Total Dead and Missing'].sum()
    fifth = df.loc[df['Reported Year'] == 2018, 'Total Dead and Missing'].sum()
    fourth = df.loc[df['Reported Year'] == 2017, 'Total Dead and Missing'].sum()
    third = df.loc[df['Reported Year'] == 2016, 'Total Dead and Missing'].sum()
    second = df.loc[df['Reported Year'] == 2015, 'Total Dead and Missing'].sum()
    first = df.loc[df['Reported Year'] == 2014, 'Total Dead and Missing'].sum()

    print('\nTotal Dead and Missing')
    print('2014: ', end="")
    print(first)
    print('2015: ', end="")
    print(second)
    print('2016: ', end="")
    print(third)
    print('2017: ', end="")
    print(fourth)
    print('2018: ', end="")
    print(fifth)
    print('2019: ', end="")
    print(sixth)

    data = {'Reported Year': ['2014', '2015', '2016', '2017', '2018', '2019'], 'Total Dead and Missing': [first, second, third, fourth, fifth, sixth]}
    dfDeadAndMissingByYear = pd.DataFrame.from_dict(data)

    x = dfDeadAndMissingByYear.loc[0:pointsToDisplay, 'Reported Year']
    y = dfDeadAndMissingByYear.loc[0:pointsToDisplay, 'Total Dead and Missing']

    figure = plt.figure('Dead+Missing')
    figure.suptitle('The Numbers of Dead and Missing Migrants Throughout the World (2014-2019)')
    #plot = plt.plot(x,y)
    plot = plt.bar(x, y, align='center', alpha=1)
    #plot = plt.scatter(x, y, s=pointSize)
    plt.xlabel('Reported Years')
    plt.ylabel('Number of Dead + Missing')
    #plt.legend(plot[:6], ['Number Dead and Missing'])

def DeadByRegion():
    figure = plt.figure('Total Dead By Region')
    #figure.suptitle('Total Migrants Who Have Died Throughout the World Arranged By Region of Death (2014-2019)')
    prob = df['Region of Incident'].value_counts()
    threshold = 30
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob['other'] = tail_prob
    print('')
    print(prob)
    plot = prob.plot(kind='bar', rot=30, fontsize=7)
    plt.xlabel('Regions')
    plt.ylabel('Total Dead by Region')
    #plt.legend(plot[:6], ['Total Dead by Region'])

def DeadByCauseOfDeath():
    figure = plt.figure('Total Dead By Cause of Death')
    figure.suptitle('Total Migrants Who Have Died While Trying to Settle Arranged By The Cause of Death (2014-2019)')
    prob = df['Cause of Death'].value_counts()
    threshold = 52
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob['other'] = tail_prob
    print('')
    print(prob)
    plot = prob.plot(kind='bar', rot=90, fontsize=5)
    plt.xlabel('Causes of Death')
    plt.ylabel('Total Dead by Cause of Death')
    #plt.legend(plot[:6], ['Total Dead by Cause of Death'])

def DeadByMigrationRoute():
    figure = plt.figure('Total Dead By Migration Route')
    figure.suptitle('Total Migrants Who Have Died While Trying to Settle Arranged By The Migration Routes (2014-2019)')
    prob = df['Migration Route'].value_counts()
    threshold = 16
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob['other'] = tail_prob
    print('')
    print(prob)
    plot = prob.plot(kind='bar', rot=20, fontsize=7)
    plt.xlabel('Migration Routes')
    plt.ylabel('Total Dead by Migration Route')
    #plt.legend(plot[:6], ['Total Dead by Migration Route'])

def DeadBySexAndAge():
    df.pivot_table(['Number of Males','Number of Females','Number of Children'], index='Reported Year', aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum}).plot(kind='bar', rot=0)
    plt.gcf().canvas.set_window_title('Total Dead By Sex And Age') #sets window title
    plt.xlabel('Reported Year')
    plt.ylabel('Total Dead')
    plt.title('Total Migrants Who Have Died While Trying to Settle Arranged By Their Sex And Age (2014-2019)') #sets graph title

def wordCloudGenerate(column, backgroundColor): # make sure the columns used don't have blank cells; ^ .fillna() methods
    allForCertainSeries = ' '.join(df[column].str.lower())
    figure = plt.figure('WordCloud by ' + column)
    df[column].iloc[0]

    wordcloud = WordCloud(background_color=backgroundColor).generate(allForCertainSeries)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

def mapOfIncidents(_projection):
    figure = plt.figure('Map of Incidents')
    figure.suptitle('6228 Migration Incidents Across the World (2014-2019)\nMade with Basemap Using Each Incident Location')

    df['Lat'], df['Lon'] = df['Location Coordinates'].str.split(', ').str
    df.Lat = df.Lat.astype(float)
    df.Lon = df.Lon.astype(float)

    lat = df['Lat'][:]
    lon = df['Lon'][:]
    #lat = lat.dropna()
    #lon = lon.dropna()
    lat = np.array(lat)
    lon = np.array(lon)

    map = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=80.,
                  resolution='l',projection=_projection,
                  lat_0=40.0,lon_0=-20.0,lat_ts=20.0)
    #map.drawmapboundary(fill_color='aqua')
    #map.fillcontinents(color='white',lake_color='aqua')
    map.drawcoastlines()
    #map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0]) # latitudinal lines and labels
    #map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1]) # longitudinal lines and labels
    x,y = map(lon,lat)
    pointSize = 1 # this number refers to the size of a point on the graph
    map.scatter(x,y,pointSize,marker='.',color='r')

#print(df.loc[df['Reported Month'] == 'Jan', 'Reported Date':'Number Dead'].iloc[0:57])

Dead()
Survivors()
Missing()
DeadAndMissing()
DeadByRegion()
DeadByCauseOfDeath()
DeadByMigrationRoute()
DeadBySexAndAge()
wordCloudGenerate('Cause of Death', 'white')
wordCloudGenerate('Region of Incident', 'white')
wordCloudGenerate('Migration Route', 'white')
mapOfIncidents('merc')

plt.show()

input() #need this for displaying multiple matplotlib windows
