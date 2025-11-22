# import json
# import os
# from datetime import datetime
# #import requests
# import pandas as pd
# from typing import Dict, List, Optional
# import math

import pandas as pd
from datetime import datetime
from typing import Dict, Any

class TickSightingsAnalyser:
    def __init__(self, file_path: str): # save file path and call load_data
        self.file_path = file_path
        self.df = self.load_data()
        # frontend create api point (url) here and dataset resource_id

    def load_data(self) -> pd.DataFrame: # load data from excel file and parse dates, expected to return DataFrame
        self.df = pd.read_excel(self.file_path)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        return self.df
    
    def filter_by_location(self, location: str) -> pd.DataFrame: # filter sightings by location, case insensitive
        return self.df[self.df["location"].str.contains(location, case=False, na=False)]

    def filter_by_date_range(self, start: str, end: str) -> pd.DataFrame: # filter sightings by date range
        start_dt = pd.to_datetime(start) # accepts july 1, 2023 or 2023-07-01 or 01/07/2023 format
        end_dt = pd.to_datetime(end)
        return self.df[(self.df["date"] >= start_dt) & (self.df["date"] <= end_dt)]
    
    def calculate_totals(self) -> Dict[str, Any]:
        totals: Dict[str, Any] = {
            "sightings_per_region": self.df["location"].value_counts().to_dict(),
            "sightings_per_species": self.df["species"].value_counts().to_dict(),
            "sightings_per_month": self.df["date"].dt.to_period("M").value_counts().sort_index().to_dict(),
            "grand_total": len(self.df),
        }
        return totals
    
    def show_sighting_per_region(self) -> None:
        return self.df["location"].value_counts().rename_axis(None).to_string()
    
    def show_sightings_per_species(self) -> None:
        return self.df["species"].value_counts().rename_axis(None).to_string()
    
    def show_sightings_per_month(self) -> None:
        return self.df["date"].dt.to_period("M").value_counts().sort_index().rename_axis(None).to_string()
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df

t = TickSightingsAnalyser('Tick Sightings.xlsx')
print(t.filter_by_location('Manchester'))
# print()
# print(t.filter_by_date_range('2023-01-01', '2023-06-30'))
# print()
# print(t.calculate_totals())
print("Ticks per location:")
print(t.show_sighting_per_region())
print()
print("Ticks per species:")
print(t.show_sightings_per_species())
print()
print("Ticks per month:")
print(t.show_sightings_per_month())