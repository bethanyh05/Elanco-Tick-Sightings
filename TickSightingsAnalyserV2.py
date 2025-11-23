import pandas as pd
from datetime import datetime
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from prophet import Prophet - problem with prophet not working 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
        self.df = self.fetch_tick_sightings()
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

    def forecast_tick_activity(self, months_ahead: int = 12):
        monthly_counts = self.df['date'].dt.to_period("M").value_counts().sort_index()
        ts = monthly_counts.reset_index()
        ts.columns = ['ds', 'y']
        ts['ds'] = ts['ds'].dt.to_timestamp()

        # Fit Holt-Winters model
        model = ExponentialSmoothing(ts['y'], trend='add', seasonal='add', seasonal_periods=12)
        #model = ExponentialSmoothing(ts['y'], trend='add', seasonal=None)
        fit = model.fit()

        forecast_values = fit.forecast(months_ahead)

        forecast = pd.DataFrame({
            'ds': pd.date_range(ts['ds'].iloc[-1] + pd.offsets.MonthEnd(),
                                periods=months_ahead, freq='ME'),
            'yhat': forecast_values
        })

        ts = ts.rename(columns={'ds': 'month', 'y': 'sightings'})
        forecast = forecast.rename(columns={'ds': 'month', 'yhat': 'future sightings'})
        forecast['future sightings'] = forecast['future sightings'].round(0).astype(int)

        return ts, forecast

# region Forecasting with prophet
    # def forecast_tick_activity(self, months_ahead: int = 6): #per month
    #     monthly_counts = self.df['date'].dt.to_period("M").value_counts().sort_index()
    #     ts = monthly_counts.reset_index()
    #     ts.columns = ['ds', 'y']  # Prophet requires these names

    #     ts['ds'] = ts['ds'].dt.to_timestamp()

    #     model = Prophet()
    #     model.fit(ts)

    #     future = model.make_future_dataframe(periods=months_ahead, freq='M')
    #     forecast = model.predict(future)

    #     return ts, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# endregion
#  
    def plot_forecast(self, ts: pd.DataFrame, forecast: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.scatter(ts['month'], ts['sightings'], color='black', label='Actual Data')
        plt.plot(forecast['month'], forecast['future sightings'], color='blue', label='Forecast')
        plt.xlabel("Date")
        plt.ylabel("Tick Sightings")
        plt.title("Tick Sightings Forecast (Holt-Winters)")
        plt.legend()
        plt.show()

    def forecast_location_trend(self, location: str, years_ahead: int = 5):
        loc_df = self.filter_by_location(location)
        if loc_df.empty:
            print(f"No data found for {location}")
            return pd.DataFrame()

        # Aggregate yearly counts
        yearly_counts = (
            loc_df['date'].dt.to_period("Y")
            .value_counts()
            .sort_index()
            .to_timestamp(how="end")
        )

        yearly_counts.index = yearly_counts.index.normalize()
        yearly_counts = yearly_counts.asfreq("YE", fill_value=0)

        #print(f"Yearly counts for {location}:\n", yearly_counts.head()) # test print 5

        # Fit Holt-Winters
        model = ExponentialSmoothing(yearly_counts, trend='add', seasonal=None)
        fit = model.fit()
        forecast_values = fit.forecast(years_ahead)

        forecast_df = pd.DataFrame({
            'date': pd.date_range(yearly_counts.index[-1] + pd.offsets.YearEnd(),
            periods=years_ahead, freq='YE'),
            'future sightings': forecast_values.round(0).astype(int)
        }).reset_index(drop=True)

        return yearly_counts, forecast_df

    def plot_location_forecast(self, location: str, six_month_counts: pd.Series, forecast_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(six_month_counts.index, six_month_counts.values, marker='o', color='black', label='Historical')
        plt.plot(forecast_df['date'], forecast_df['future sightings'], marker='o', color='blue', label='Forecast')
        plt.xlabel("6-Month Period")
        plt.ylabel("Tick Sightings")
        plt.title(f"Tick Sightings Trend (6-Month Intervals) for {location}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


t = TickSightingsAnalyser('Tick Sightings.xlsx')

#FILTER TESTS
print("Location Filter Test:")
print(t.filter_by_location('Manchester'))
print()

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

#AI/ML TESTS
#FORECASTING TEST
# ts, forecast = t.forecast_tick_activity(months_ahead=24)
# print("Historical monthly counts:")
# print(ts) # ts.head() first 5 rows

# print("\nForecasted values:")
# print(forecast)

# t.plot_forecast(ts, forecast)

#CLUSTERING TEST
print("\nLocation clusters and forecasts:")
# Call forecast for Manchester, 5 years ahead
yearly_counts, forecast_df = t.forecast_location_trend("Manchester", years_ahead=5)

# Print results
print("Historical yearly counts:")
print(yearly_counts.reset_index())
print(forecast_df)

# print("\nForecasted values:")
# print(forecast_df)
# t.plot_location_forecast("Manchester", yearly_counts, forecast_df)