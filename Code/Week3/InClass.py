# Data Management - Exercise
# %%
# Exercise 1
import pandas as pd

delays18 = pd.read_csv('https://raw.githubusercontent.com/nickdcox/learn-airline-delays/main/delays_2018.csv')
delays19 = pd.read_csv('https://raw.githubusercontent.com/nickdcox/learn-airline-delays/main/delays_2019.csv')
delays = pd.concat([delays18, delays19], ignore_index=True)

# How to change data type !
delays['date'] = pd.to_datetime(delays.date, format='%Y-%m')

# Remove rows based on criteria ( by keeping stuff )
delays_new = delays[(delays.date >= '2018-01') &
                    (delays.date <= '2019-12') &
                    delays.arr_flights.notnull() &
                    delays.carrier.notnull() &
                    delays.airport.notnull()]

# Remove rows based on criteria ( by dropping stuff )
delays_new2 = delays.drop(delays[delays.date >= '2018-01'], axis=1)
print(len(delays_new2))

# See lists of things that contains something
airports_in_TN = delays_new[(delays_new.airport_name.str.contains('TN:', na=False))]

###
airport_coordinates = pd.read_csv('https://raw.githubusercontent.com/nickdcox/learn-airline-delays/main/airport_coordinates.csv')

# remove duplicates
airports = delays[['airport','airport_name']].drop_duplicates().reset_index(drop=True)
print(airports)

# merge  is similar to joining tables in SQL
airports = airports.merge(airport_coordinates, on='airport')

# airports = airports.merge(airport_coordinates, left_on='airport', right_on='airport')
print(airports)

# B_airports = airports[airports.airport.str.contains('^B.*$', regex=True, na=False)]

B_airports = airports[airports.airport.str.startswith('B',na=False)]
print(B_airports)

print(B_airports)

# %%
# Exercise 2


