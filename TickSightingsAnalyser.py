import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, Any

class TickSightingsAnalyser:
    def __init__(self, file_path: str): # save file path and call load_data
        self.file_path = file_path
        self.df = self.load_data()
        # frontend create api point (url) here and dataset resource_id
    
    def fetch_tick_sightings(self) -> pd.DataFrame:
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded {len(self.df)} tick sightings")
            return self.df
        except Exception as e:
            print(f"Error loading tick sightings: {e}")
            return pd.DataFrame()
    
    def load_data(self) -> pd.DataFrame: # load data from excel file and parse dates, expected to return DataFrame
        self.fetch_tick_sightings()
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        hour = self.df["date"].dt.hour
        conditions = [
            (hour >= 5) & (hour < 12),
            (hour >= 12) & (hour < 17),
            (hour >= 17) & (hour < 21),
            (hour < 5) | (hour >= 21)
        ]
        choices = ["Morning", "Afternoon", "Evening", "Night"]
        self.df["timeOfDay"] = np.select(conditions, choices, default="Unknown")

        return self.df

# region Data Filtering
    def filter_by_location(self, location: str) -> pd.DataFrame: # filter sightings by location, case insensitive
        return self.df[self.df["location"].str.contains(location, case=False, na=False)]
    
    def filter_by_date_range(self, start: str, end: str) -> pd.DataFrame: # filter sightings by date range
        start_dt = pd.to_datetime(start) # accepts july 1, 2023 or 2023-07-01 or 01/07/2023 format
        end_dt = pd.to_datetime(end)
        return self.df[(self.df["date"] >= start_dt) & (self.df["date"] <= end_dt)]
    
    def filter_by_time_of_day(self, time: str ) -> pd.DataFrame:
        try:
            #filter time ranges
            if "-" in time:
                start_str, end_str = time.split("-")
                start_hour = datetime.strptime(start_str.strip(), "%H:%M").hour
                end_hour = datetime.strptime(end_str.strip(), "%H:%M").hour
                return self.df[self.df["date"].dt.hour.between(start_hour, end_hour - 1)]
            
            #filter specific time
            hour = datetime.strptime(time, '%H:%M').hour
            return self.df[self.df["date"].dt.hour == hour]
        
        except ValueError:
            #filter by time of day category
            categories = ["Morning", "Afternoon", "Evening", "Night"]
            if time in categories:
                return self.df[self.df["timeOfDay"].str.contains(time, case=False, na=False)]
            else:
                print(f"Invalid input: {time}. Must be a category or HH:MM time.")
                return pd.DataFrame()
# endregion

# region Data Reporting
            
    def show_sighting_per_region(self) -> None:
        return self.df["location"].value_counts().rename_axis(None).to_string()
    
    def show_sightings_per_species(self) -> None:
        return self.df["species"].value_counts().rename_axis(None).to_string()
    
    def show_sightings_per_month(self) -> None:
        return self.df["date"].dt.to_period("M").value_counts().sort_index().rename_axis(None).to_string()
    
    def show_sightings_per_week(self) -> None:
        return self.df["date"].dt.to_period("W").value_counts().sort_index().rename_axis(None).to_string()
# endregion

t = TickSightingsAnalyser('Tick Sightings.xlsx')

#FILTER TESTS
#print("Location Filter Test:")
#print(t.filter_by_location('Manchester'))
#print()

#print("Date Range Filter Test:")
#print(t.filter_by_date_range('2023-01-01', '2023-06-30'))
#print()

#print("Time of Day Filter Test:")
#print(t.filter_by_time_of_day('Morning'))
#print(t.filter_by_time_of_day("06:40"))
#print(t.filter_by_time_of_day("14:00-18:00"))

#GROUPING TESTS
# print("Ticks per location:")
# print(t.show_sighting_per_region())
# print()
# print("Ticks per species:")
# print(t.show_sightings_per_species())
# print()
# print("Ticks per month:")
# print(t.show_sightings_per_month())
#print("Ticks per week:")
#print(t.show_sightings_per_week())
