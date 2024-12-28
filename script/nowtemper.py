import requests
# to get data from website
file = requests.get("https://weather.com/weather/today/l/e5d48f32e8ccf2d57fe8ec97c002da31a283ea2400adfac3e0a267cad788ae8f?unit=m")

# import Beautifulsoup for scraping the data 
from bs4 import BeautifulSoup
soup = BeautifulSoup(file.content, "html.parser")
list =[]
# find all table with class-"twc-table"
content = soup.find("div", {"id":"todayDetails"})
for items in content:
    dict = {}
    try:
        todaydetails = items.find("div", {"id":"todayDetails"})
                # assign value to given key 
        print(items.find("div", {"data-testid":"TodaysDetailsHeader"}).find("span",{"data-testid":"TemperatureValue"}).text)
        
        dict["temp"]= items.find("div", {"data-testid":"TodaysDetailsHeader"}).find("span",{"data-testid":"TemperatureValue"}).text.replace('°','')
        print(dict["temp"])
        dict["pressure"]= items.find("span", {"class":"Pressure--pressureWrapper--3SCLm"}).find_all("span")[0].text
        dict["wind"]= items.find("span", {"class":"Wind--windWrapper--3Ly7c undefined"}).find_all("span")[1].text
        dict["humidity"]= items.find_all("div", {"data-testid":"wxData"})[2].text.replace('%','')
        dict["dew_point"]= items.find_all("div", {"data-testid":"wxData"})[3].text.replace('°','')
    except:  
             # assign None values if no items are there with specified class
        dict["temp"]="None"
        dict["pressure"]="None"
        dict["wind"]="None"
        dict["humidity"]="None"
        dict["dew_point"]="None"

    # append dictionary values to the list
    list.append(dict)

import pandas as pd
convert = pd.DataFrame(list)
convert.to_csv("output.csv")


# read csv file using pandas
a = pd.read_csv("output.csv")
print(a)
