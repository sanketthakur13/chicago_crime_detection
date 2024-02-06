import numpy as np
import pickle
import streamlit as st
import base64
import folium
#
#
# #####################code for model############################
import pandas as pd
from wordcloud import WordCloud
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
#
df = pd.read_csv('chicago_crime_ml.csv')
loaded_model = pickle.load(open('chicago_crime_new.sav','rb'))

# Dictionary to map string labels to encoded numbers for each column
label_mappings = { 'LocationDescription':
                       {'ABANDONED BUILDING': 70,
                             'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA': 55,
                             'AIRPORT BUILDING NON-TERMINAL - SECURE AREA': 38,
                             'AIRPORT EXTERIOR - NON-SECURE AREA': 74,
                             'AIRPORT EXTERIOR - SECURE AREA': 69,
                             'AIRPORT PARKING LOT': 51,
                             'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA': 59,
                             'AIRPORT VENDING ESTABLISHMENT': 60,
                             'AIRPORT/AIRCRAFT': 57,
                             'ALLEY': 33,
                             'APARTMENT': 66,
                             'APPLIANCE STORE': 28,
                             'ATHLETIC CLUB': 9,
                             'BANK': 73,
                             'BAR OR TAVERN': 67,
                             'BARBERSHOP': 54,
                             'BOWLING ALLEY': 21,
                             'BRIDGE': 68,
                             'CAR WASH': 20,
                             'CHA APARTMENT': 10,
                             'CHA PARKING LOT/GROUNDS': 41,
                             'CHURCH/SYNAGOGUE/PLACE OF WORSHIP': 71,
                             'CLEANING STORE': 32,
                             'COLLEGE/UNIVERSITY GROUNDS': 61,
                             'COLLEGE/UNIVERSITY RESIDENCE HALL': 40,
                             'COMMERCIAL / BUSINESS OFFICE': 65,
                             'CONSTRUCTION SITE': 12,
                             'CONVENIENCE STORE': 18,
                             'CTA GARAGE / OTHER PROPERTY': 48,
                             'CTA TRAIN': 56,
                             'CURRENCY EXCHANGE': 43,
                             'DAY CARE CENTER': 50,
                             'DEPARTMENT STORE': 46,
                             'DRIVEWAY - RESIDENTIAL': 25,
                             'DRUG STORE': 53,
                             'FACTORY/MANUFACTURING BUILDING': 34,
                             'FIRE STATION': 58,
                             'FOREST PRESERVE': 62,
                             'GAS STATION': 26,
                             'GOVERNMENT BUILDING/PROPERTY': 23,
                             'GROCERY FOOD STORE': 63,
                             'HIGHWAY/EXPRESSWAY': 39,
                             'HOSPITAL BUILDING/GROUNDS': 72,
                             'HOTEL/MOTEL': 42,
                             'JAIL / LOCK-UP FACILITY': 64,
                             'LAKEFRONT/WATERFRONT/RIVERBANK': 13,
                             'LIBRARY': 15,
                             'MEDICAL/DENTAL OFFICE': 14,
                             'MOVIE HOUSE/THEATER': 35,
                             'NEWSSTAND': 22,
                             'NURSING HOME/RETIREMENT HOME': 76,
                             'OTHER': 37,
                             'OTHER COMMERCIAL TRANSPORTATION': 8,
                             'OTHER RAILROAD PROP / TRAIN DEPOT': 19,
                             'PARK PROPERTY': 52,
                             'PARKING LOT/GARAGE(NON.RESID.)': 30,
                             'POLICE FACILITY/VEH PARKING LOT': 16,
                             'RESIDENCE': 36,
                             'RESIDENCE PORCH/HALLWAY': 45,
                             'RESIDENCE-GARAGE': 75,
                             'RESIDENTIAL YARD (FRONT/BACK)': 31,
                             'RESTAURANT': 7,
                             'SAVINGS AND LOAN': 5,
                             'SCHOOL, PRIVATE, BUILDING': 0,
                             'SCHOOL, PRIVATE, GROUNDS': 6,
                             'SCHOOL, PUBLIC, BUILDING': 3,
                             'SCHOOL, PUBLIC, GROUNDS': 47,
                             'SIDEWALK': 17,
                             'SMALL RETAIL STORE': 4,
                             'SPORTS ARENA/STADIUM': 1,
                             'STREET': 24,
                             'TAVERN/LIQUOR STORE': 2,
                             'TAXICAB': 29,
                             'VACANT LOT/LAND': 27,
                             'VEHICLE NON-COMMERCIAL': 44,
                             'VEHICLE-COMMERCIAL': 49,
                             'WAREHOUSE': 11},
    'Domestic': {'No': 0, 'Yes': 1},
    'CommunityArea': {'(The) Loop': 29,
                     'Albany Park': 62,
                     'Archer Heights': 4,
                     'Armour Square': 54,
                     'Ashburn': 15,
                     'Auburn Gresham': 23,
                     'Austin': 5,
                     'Avalon Park': 13,
                     'Avondale': 60,
                     'Belmont Cragin': 58,
                     'Beverly': 67,
                     'Bridgeport': 33,
                     'Brighton Park': 66,
                     'Burnside': 35,
                     'Calumet Heights': 18,
                     'Chatham': 9,
                     'Chicago Lawn': 14,
                     'Clearing': 46,
                     'Douglas': 12,
                     'Dunning': 16,
                     'East Garfield Park': 1,
                     'East Side': 0,
                     'Edgewater': 7,
                     'Englewood': 26,
                     'Forest Glen': 28,
                     'Fuller Park': 6,
                     'Gage Park': 49,
                     'Garfield Ridge': 3,
                     'Grand Boulevard': 61,
                     'Greater Grand Crossing': 45,
                     'Hegewisch': 19,
                     'Hermosa': 34,
                     'Humboldt Park': 57,
                     'Hyde Park': 40,
                     'Irving Park': 20,
                     'Jefferson Park': 36,
                     'Kenwood': 38,
                     'Logan Square': 32,
                     'Lower West Side': 8,
                     'McKinley Park': 44,
                     'Montclare': 52,
                     'Morgan Park': 31,
                     'Mount Greenwood': 55,
                     'Near South Side': 17,
                     'Near West Side': 22,
                     'New City': 65,
                     'North Lawndale': 21,
                     'North Park': 64,
                     'Norwood Park': 27,
                     "O'Hare": 59,
                     'Oakland': 47,
                     'Portage Park': 37,
                     'Pullman': 51,
                     'Riverdale': 41,
                     'Roseland': 2,
                     'South Chicago': 24,
                     'South Deering': 53,
                     'South Lawndale': 63,
                     'South Shore': 11,
                     'Washington Heights': 50,
                     'Washington Park': 25,
                     'West Elsdon': 10,
                     'West Englewood': 56,
                     'West Garfield Park': 39,
                     'West Lawn': 48,
                     'West Pullman': 42,
                     'West Town': 43,
                     'Woodlawn': 30},
    'District': {'Albany Park': 9,
                     'Austin': 8,
                     'Calumet': 6,
                     'Central': 4,
                     'Chicago Lawn': 19,
                     'Deering': 16,
                     'Englewood': 18,
                     'Grand Central': 15,
                     'Grand Crossing': 11,
                     'Gresham': 21,
                     'Harrison': 3,
                     'Jefferson Park': 5,
                     'Lincoln': 20,
                     'Morgan Park': 7,
                     'Near North': 17,
                     'Near West': 10,
                     'New West': 0,
                     'Ogden': 14,
                     'Shakespeare': 2,
                     'South Chicago': 13,
                     'Unknown': 1,
                     'Wentworth': 12},
    'Beat': {'1011': 243,
             '1012': 244,
             '1013': 241,
             '1014': 237,
             '1021': 240,
             '1022': 248,
             '1023': 247,
             '1024': 228,
             '1031': 246,
             '1032': 245,
             '1033': 242,
             '1034': 238,
             '111': 239,
             '1111': 210,
             '1112': 209,
             '1113': 205,
             '1114': 211,
             '1115': 204,
             '112': 208,
             '1121': 254,
             '1122': 213,
             '1123': 214,
             '1124': 212,
             '1125': 215,
             '113': 275,
             '1131': 219,
             '1132': 206,
             '1133': 207,
             '1134': 148,
             '1135': 259,
             '114': 260,
             '121': 256,
             '1211': 262,
             '1212': 257,
             '1213': 251,
             '1214': 263,
             '1215': 253,
             '122': 252,
             '1221': 258,
             '1222': 250,
             '1223': 249,
             '1224': 255,
             '1225': 261,
             '123': 174,
             '1231': 34,
             '1232': 54,
             '1233': 74,
             '1234': 70,
             '1235': 57,
             '124': 58,
             '130': 71,
             '131': 56,
             '1311': 75,
             '1312': 53,
             '1313': 52,
             '132': 68,
             '1322': 61,
             '1323': 90,
             '1324': 85,
             '133': 91,
             '1331': 89,
             '1332': 142,
             '1333': 138,
             '134': 143,
             '1411': 140,
             '1412': 141,
             '1413': 190,
             '1414': 193,
             '1421': 194,
             '1422': 195,
             '1423': 201,
             '1424': 192,
             '1431': 189,
             '1432': 218,
             '1433': 222,
             '1434': 217,
             '1511': 166,
             '1512': 178,
             '1513': 170,
             '1522': 3,
             '1523': 28,
             '1524': 27,
             '1531': 0,
             '1532': 1,
             '1533': 6,
             '1611': 4,
             '1612': 2,
             '1613': 26,
             '1614': 25,
             '1621': 5,
             '1622': 7,
             '1623': 29,
             '1624': 269,
             '1631': 284,
             '1632': 279,
             '1633': 282,
             '1634': 280,
             '1651': 281,
             '1654': 273,
             '1711': 274,
             '1712': 271,
             '1713': 272,
             '1722': 270,
             '1723': 278,
             '1724': 277,
             '1731': 276,
             '1732': 104,
             '1733': 102,
             '1811': 103,
             '1812': 100,
             '1813': 101,
             '1814': 30,
             '1821': 12,
             '1822': 49,
             '1823': 24,
             '1824': 18,
             '1831': 51,
             '1832': 55,
             '1833': 37,
             '1834': 43,
             '1911': 234,
             '1912': 229,
             '1913': 231,
             '1922': 162,
             '1923': 230,
             '1924': 161,
             '1931': 216,
             '1932': 220,
             '1933': 286,
             '2011': 283,
             '2012': 150,
             '2013': 164,
             '2022': 146,
             '2023': 160,
             '2024': 152,
             '2031': 156,
             '2032': 15,
             '2033': 200,
             '211': 83,
             '2111': 98,
             '2112': 81,
             '2113': 264,
             '212': 199,
             '2122': 76,
             '2123': 77,
             '2124': 191,
             '213': 79,
             '2131': 84,
             '2132': 78,
             '2133': 82,
             '214': 13,
             '215': 80,
             '221': 114,
             '2211': 267,
             '2212': 115,
             '2213': 203,
             '222': 198,
             '2221': 64,
             '2222': 60,
             '2223': 167,
             '223': 154,
             '2232': 69,
             '2233': 268,
             '2234': 223,
             '224': 137,
             '225': 134,
             '231': 158,
             '2311': 182,
             '2312': 95,
             '2313': 157,
             '232': 20,
             '2322': 39,
             '2323': 125,
             '2324': 188,
             '233': 292,
             '2331': 159,
             '2332': 233,
             '2333': 224,
             '234': 41,
             '235': 9,
             '2411': 131,
             '2412': 127,
             '2413': 32,
             '2422': 124,
             '2423': 144,
             '2424': 119,
             '2431': 285,
             '2432': 128,
             '2433': 163,
             '2511': 149,
             '2512': 86,
             '2513': 136,
             '2514': 33,
             '2515': 180,
             '2521': 175,
             '2522': 291,
             '2523': 67,
             '2524': 120,
             '2525': 65,
             '2531': 266,
             '2532': 106,
             '2533': 97,
             '2534': 183,
             '2535': 287,
             '311': 197,
             '312': 46,
             '313': 235,
             '314': 187,
             '321': 185,
             '322': 221,
             '323': 99,
             '324': 44,
             '331': 8,
             '332': 168,
             '333': 202,
             '334': 186,
             '411': 236,
             '412': 108,
             '413': 155,
             '414': 116,
             '421': 22,
             '422': 14,
             '423': 117,
             '424': 176,
             '431': 139,
             '432': 145,
             '433': 121,
             '434': 133,
             '511': 113,
             '512': 96,
             '513': 181,
             '522': 62,
             '523': 17,
             '524': 289,
             '531': 122,
             '532': 73,
             '533': 66,
             '611': 40,
             '612': 92,
             '613': 129,
             '614': 172,
             '621': 16,
             '622': 45,
             '623': 225,
             '624': 196,
             '631': 88,
             '632': 11,
             '633': 105,
             '634': 10,
             '711': 227,
             '712': 290,
             '713': 19,
             '714': 118,
             '715': 226,
             '722': 169,
             '723': 132,
             '724': 107,
             '725': 293,
             '726': 59,
             '731': 232,
             '732': 153,
             '733': 123,
             '734': 126,
             '735': 147,
             '811': 109,
             '812': 173,
             '813': 288,
             '814': 23,
             '815': 72,
             '821': 184,
             '822': 112,
             '823': 21,
             '824': 94,
             '825': 265,
             '831': 130,
             '832': 135,
             '833': 171,
             '834': 177,
             '835': 93,
             '911': 87,
             '912': 110,
             '913': 111,
             '914': 63,
             '915': 36,
             '921': 35,
             '922': 38,
             '923': 48,
             '924': 47,
             '925': 42,
             '931': 165,
             '932': 151,
             '933': 179,
             '934': 50,
             '935': 31}}

def crime_prediction(input_data):
    prediction = loaded_model.predict([input_data])
    return prediction[0]
    print(prediction)
    if (prediction[0] == 0):
        return 'The suspect has not been arrested'
    else:
        return 'The suspect has been arrested'


# ##############################code for the web-app#################################

def main():


    st.markdown("<h1 style='color: white;'>Crime Detection in Chicago</h1>", unsafe_allow_html=True)


    # Setting the background Dirty Harry
    def set_background(png_file_path, size="100% 100%", repeat="no-repeat", position="center"):
        # Function to read the PNG file and encode it into base64
        with open(png_file_path, "rb") as file:
            bin_str = base64.b64encode(file.read()).decode()

        # CSS style string to set the background image with specified properties
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: {size};
            background-repeat: {repeat};
            background-position: {position};
        }}
        </style>
        '''

        # Apply the CSS style to the Streamlit app
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Call set_background function with the file path and specified arguments
    set_background(r'clint-eastwood-dirty-harry.gif', size="cover", repeat="no-repeat",
                   position="center")

    # getting the data
    location_description = st.selectbox('What is the Location of Crime?',('STREET', 'RESIDENTIAL YARD (FRONT/BACK)', 'GAS STATION', 'PARKING LOT/GARAGE(NON.RESID.)', 'VEHICLE NON-COMMERCIAL', 'CTA GARAGE / OTHER PROPERTY', 'RESIDENCE-GARAGE', 'OTHER', 'ALLEY', 'SPORTS ARENA/STADIUM', 'VACANT LOT/LAND', 'RESIDENCE', 'SCHOOL, PUBLIC, GROUNDS', 'DRIVEWAY - RESIDENTIAL', 'POLICE FACILITY/VEH PARKING LOT', 'SIDEWALK', 'APARTMENT', 'VEHICLE-COMMERCIAL', 'AIRPORT VENDING ESTABLISHMENT', 'BAR OR TAVERN', 'PARK PROPERTY', 'HIGHWAY/EXPRESSWAY', 'COLLEGE/UNIVERSITY GROUNDS', 'SMALL RETAIL STORE', 'CAR WASH', 'FACTORY/MANUFACTURING BUILDING', 'RESTAURANT', 'FIRE STATION', 'CHA PARKING LOT/GROUNDS', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'CTA TRAIN', 'GROCERY FOOD STORE', 'AIRPORT PARKING LOT', 'MOVIE HOUSE/THEATER', 'TAVERN/LIQUOR STORE', 'GOVERNMENT BUILDING/PROPERTY', 'NURSING HOME/RETIREMENT HOME', 'AIRPORT/AIRCRAFT', 'HOTEL/MOTEL', 'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'CONSTRUCTION SITE', 'COMMERCIAL / BUSINESS OFFICE', 'SCHOOL, PUBLIC, BUILDING', 'DEPARTMENT STORE', 'WAREHOUSE', 'HOSPITAL BUILDING/GROUNDS', 'RESIDENCE PORCH/HALLWAY', 'AIRPORT EXTERIOR - SECURE AREA', 'TAXICAB', 'CHA APARTMENT', 'SCHOOL, PRIVATE, GROUNDS', 'CONVENIENCE STORE', 'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'BANK', 'OTHER RAILROAD PROP / TRAIN DEPOT', 'MEDICAL/DENTAL OFFICE', 'DRUG STORE', 'NEWSSTAND', 'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION', 'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'LIBRARY', 'ATHLETIC CLUB', 'FOREST PRESERVE', 'BRIDGE', 'SAVINGS AND LOAN', 'DAY CARE CENTER', 'ABANDONED BUILDING', 'CURRENCY EXCHANGE', 'SCHOOL, PRIVATE, BUILDING', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'BARBERSHOP', 'BOWLING ALLEY', 'JAIL / LOCK-UP FACILITY', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'APPLIANCE STORE', 'ANIMAL HOSPITAL', 'CLEANING STORE'))
    st.write('<span style="color:white">You selected:</span>', location_description, unsafe_allow_html=True)

    domestic = st.selectbox('Is the criminal domestic or not?',
        ('Yes','No'))
    st.write('<span style="color:white">You selected:</span>', domestic, unsafe_allow_html=True)


    district = st.selectbox('From which district does the criminal belong?',
                            ('Albany Park', 'Austin', 'Calumet', 'Central', 'Chicago Lawn', 'Deering', 'Englewood', 'Grand Central', 'Grand Crossing', 'Gresham', 'Harrison', 'Jefferson Park', 'Lincoln', 'Morgan Park', 'Near North', 'Near West', 'New West', 'Ogden', 'Shakespeare', 'South Chicago', 'Unknown', 'Wentworth'))
    st.write('<span style="color:white">You selected:</span>', district, unsafe_allow_html=True)

    community_area = st.selectbox('Which Community Area did the crime take place?',
                            ('(The) Loop', 'Albany Park', 'Archer Heights', 'Armour Square', 'Ashburn', 'Auburn Gresham', 'Austin', 'Avalon Park', 'Avondale', 'Belmont Cragin', 'Beverly', 'Bridgeport', 'Brighton Park', 'Burnside', 'Calumet Heights', 'Chatham', 'Chicago Lawn', 'Clearing', 'Douglas', 'Dunning', 'East Garfield Park', 'East Side', 'Edgewater', 'Englewood', 'Forest Glen', 'Fuller Park', 'Gage Park', 'Garfield Ridge', 'Grand Boulevard', 'Greater Grand Crossing', 'Hegewisch', 'Hermosa', 'Humboldt Park', 'Hyde Park', 'Irving Park', 'Jefferson Park', 'Kenwood', 'Logan Square', 'Lower West Side', 'McKinley Park', 'Montclare', 'Morgan Park', 'Mount Greenwood', 'Near South Side', 'Near West Side', 'New City', 'North Lawndale', 'North Park', 'Norwood Park', "O'Hare", 'Oakland', 'Portage Park', 'Pullman', 'Riverdale', 'Roseland', 'South Chicago', 'South Deering', 'South Lawndale', 'South Shore', 'Washington Heights', 'Washington Park', 'West Elsdon', 'West Englewood', 'West Garfield Park', 'West Lawn', 'West Pullman', 'West Town', 'Woodlawn'))
    st.write('<span style="color:white">You selected:</span>', community_area, unsafe_allow_html=True)

    beat = st.selectbox('In which Beat did the crime take place?',
                                  ('1011', '1012', '1013', '1014', '1021', '1022', '1023', '1024', '1031', '1032', '1033', '1034', '111', '1111', '1112', '1113', '1114', '1115', '112', '1121', '1122', '1123', '1124', '1125', '113', '1131', '1132', '1133', '1134', '1135', '114', '121', '1211', '1212', '1213', '1214', '1215', '122', '1221', '1222', '1223', '1224', '1225', '123', '1231', '1232', '1233', '1234', '1235', '124', '130', '131', '1311', '1312', '1313', '132', '1322', '1323', '1324', '133', '1331', '1332', '1333', '134', '1411', '1412', '1413', '1414', '1421', '1422', '1423', '1424', '1431', '1432', '1433', '1434', '1511', '1512', '1513', '1522', '1523', '1524', '1531', '1532', '1533', '1611', '1612', '1613', '1614', '1621', '1622', '1623', '1624', '1631', '1632', '1633', '1634', '1651', '1654', '1711', '1712', '1713', '1722', '1723', '1724', '1731', '1732', '1733', '1811', '1812', '1813', '1814', '1821', '1822', '1823', '1824', '1831', '1832', '1833', '1834', '1911', '1912', '1913', '1922', '1923', '1924', '1931', '1932', '1933', '2011', '2012', '2013', '2022', '2023', '2024', '2031', '2032', '2033', '211', '2111', '2112', '2113', '212', '2122', '2123', '2124', '213', '2131', '2132', '2133', '214', '215', '221', '2211', '2212', '2213', '222', '2221', '2222', '2223', '223', '2232', '2233', '2234', '224', '225', '231', '2311', '2312', '2313', '232', '2322', '2323', '2324', '233', '2331', '2332', '2333', '234', '235', '2411', '2412', '2413', '2422', '2423', '2424', '2431', '2432', '2433', '2511', '2512', '2513', '2514', '2515', '2521', '2522', '2523', '2524', '2525', '2531', '2532', '2533', '2534', '2535', '311', '312', '313', '314', '321', '322', '323', '324', '331', '332', '333', '334', '411', '412', '413', '414', '421', '422', '423', '424', '431', '432', '433', '434', '511', '512', '513', '522', '523', '524', '531', '532', '533', '611', '612', '613', '614', '621', '622', '623', '624', '631', '632', '633', '634', '711', '712', '713', '714', '715', '722', '723', '724', '725', '726', '731', '732', '733', '734', '735', '811', '812', '813', '814', '815', '821', '822', '823', '824', '825', '831', '832', '833', '834', '835', '911', '912', '913', '914', '915', '921', '922', '923', '924', '925', '931', '932', '933', '934', '935'))
    st.write('<span style="color:white">You selected:</span>', beat, unsafe_allow_html=True)


    # Convert selected values to their corresponding encoded numbers
    encoded_location_description = label_mappings['LocationDescription'][location_description]
    encoded_domestic = label_mappings['Domestic'][domestic]
    encoded_district = label_mappings['District'][district]
    encoded_community_area = label_mappings['CommunityArea'][community_area]
    encoded_beat = label_mappings['Beat'][beat]




    #prediction
    if st.button('Predict'):
        # Combine encoded features into a single array
        input_features = [encoded_location_description, encoded_domestic,encoded_beat, encoded_district,encoded_community_area]
        # Make predictions
        prediction = crime_prediction(input_features)

        if prediction == 0:
            st.write('Predicted result: The suspect has not been arrested')
        else:
            st.write('Predicted result: The suspect has been arrested')



        #EDA
        # Convert 'Year' to datetime type
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')

        # Group by year and count the number of True and False arrests
        arrest_counts = df.groupby([df['Year'].dt.year, 'Arrest']).size().unstack(fill_value=0)

        # Plotting
        fig, ax = plt.subplots()
        arrest_counts.plot(kind='line', marker='o', ax=ax)
        plt.title('Year-wise Arrests')
        plt.xlabel('Year')
        plt.ylabel('Number of Arrests')
        plt.legend(title='Arrest', loc='upper left')
        plt.xticks(arrest_counts.index)
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)


        # st.title("Chicago District Map")

        # # Coordinates for Chicago
        # chicago_coords = [41.8781, -87.6298]
    
        # # Create a map centered around Chicago
        # chicago_map = folium.Map(location=chicago_coords, zoom_start=10)
    
        # # Add district names to the map
        # districts = {
        #     "Central": [41.8781, -87.6298],
        #     "North Side": [41.9100, -87.6300],
        #     "South Side": [41.7700, -87.6300],
        #     "West Side": [41.8781, -87.7430]
        # }
    
        # # Add markers for each district
        # for district, coords in districts.items():
        #     folium.Marker(location=coords, popup=district).add_to(chicago_map)

        # #wordcloud
        # # Filter DataFrame to include only rows where arrest is True
        # arrest_true_df = df[df['Arrest'] == True]
        #
        # # Extract LocationDescription column
        # location_description = arrest_true_df['LocationDescription']
        #
        # # Join all location descriptions into a single string
        # text = ' '.join(location_description)
        #
        # # Generate word cloud
        # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        #
        # # Display the generated word cloud image in Streamlit
        # st.image(wordcloud.to_array(), caption='Word Cloud of Location Descriptions with True Arrests')

if __name__ == '__main__':
    main()
