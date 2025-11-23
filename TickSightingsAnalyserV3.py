import pandas as pd
from datetime import datetime
# from prophet import Prophet - problem with prophet not working
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class TickSightingsAnalyser:
    def __init__(self, file_path: str): # save file path and call load_data
        self.file_path = file_path
        self.df = self._load_data()
        # frontend create api point (url) here and dataset resource_id
    
    def _fetch_tick_sightings(self) -> pd.DataFrame:
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded {len(self.df)} tick sightings")
            return self.df
        except Exception as e:
            print(f"Error loading tick sightings: {e}")
            return pd.DataFrame()
    
    def _load_data(self) -> pd.DataFrame: # load data from excel file and parse dates, expected to return DataFrame
        self.df = self._fetch_tick_sightings()
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
    def filter_by_location(self, location: str, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        return df[df["location"].str.contains(location, case=False, na=False)]

    def filter_by_date_range(self, start: str, end: str, df: pd.DataFrame = None) -> pd.DataFrame: # filter sightings by date range
        if df is None:
            df = self.df

        start_dt = pd.to_datetime(start) # accepts july 1, 2023 or 2023-07-01 or 01/07/2023 format
        end_dt = pd.to_datetime(end)
        return self.df[(self.df["date"] >= start_dt) & (self.df["date"] <= end_dt)]

    def filter_by_time_of_day(self, time: str, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        try:
            if "-" in time:  # filter time ranges
                start_str, end_str = time.split("-")
                start_hour = datetime.strptime(start_str.strip(), "%H:%M").hour
                end_hour = datetime.strptime(end_str.strip(), "%H:%M").hour
                return df[df["date"].dt.hour.between(start_hour, end_hour - 1)]
            else:  # filter specific time
                hour = datetime.strptime(time, "%H:%M").hour
                return df[df["date"].dt.hour == hour]
        except ValueError:
            categories = ["Morning", "Afternoon", "Evening", "Night"]
            if time in categories:
                return df[df["timeOfDay"].str.contains(time, case=False, na=False)]
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
        # filter data for the specified location and add to a new dataframe
        loc_df = self.filter_by_location(location)
        if loc_df.empty:
            print(f"No data found for {location}")
            return pd.Series(dtype=int), pd.DataFrame()

        loc_df = loc_df.copy()
        loc_df['date'] = pd.to_datetime(loc_df['date'], errors='coerce')
        loc_df = loc_df.dropna(subset=['date']).set_index('date').sort_index()

        # count sightings per year, resample to ensure all years are present
        yearly = loc_df.resample('YE').size().rename('sightings').to_frame()
        if yearly.empty:
            return pd.Series(dtype=int), pd.DataFrame()
        
        # order data by year
        start = yearly.index.min()
        end = yearly.index.max()
        full_idx = pd.date_range(start=start, end=end, freq='YE')
        yearly = yearly.reindex(full_idx, fill_value=0)
        yearly.index.name = 'date'
        y = yearly['sightings'].astype(int)

        n = len(y)
        max_past = min(3, max(0, n - 2))
        df = pd.DataFrame({'year': y.index.year, 'sightings': y.values})
        for past in range(1, max_past + 1):
            df[f'past{past}'] = df['sightings'].shift(past)
        df = df.dropna().reset_index(drop=True)

        if not df.empty:
            X = df.drop(columns=['sightings']).values
            Y = df['sightings'].values   # NumPy array for training
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X, Y)
            last_counts = list(Y[-max_past:])

            # create sightings predictions
            current_year = y.index.year[-1]
            predictions = []
            while len(predictions) < years_ahead:
                current_year += 1
                features = [current_year] + last_counts[::-1]
                pred = max(0.0, model.predict([features])[0])
                predictions.append(pred)
                last_counts = (last_counts + [pred])[-max_past:]
        else:
            # predict when small amount of data (less than three years)
            years = y.index.year.values.reshape(-1, 1)
            vals = y.values
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(years, vals)
            last_year = years.flatten()[-1]
            pred_years = np.arange(last_year + 1, last_year + 1 + years_ahead).reshape(-1, 1)
            predictions = [max(0.0, p) for p in model.predict(pred_years)]

        # build past and predicted dataframe to plot later
        predictions = np.round(np.clip(np.asarray(predictions), 0, None)).astype(int)
        future_years = np.arange(y.index.year[-1] + 1, y.index.year[-1] + 1 + years_ahead)
        forecast_dates = pd.to_datetime(future_years.astype(str), format='%Y') + pd.offsets.YearEnd(0)
        forecast_df = pd.DataFrame({'date': forecast_dates, 'future sightings': predictions}).reset_index(drop=True)

        return y, forecast_df
        
    def plot_location_forecast(self, location: str, month_counts: pd.Series, forecast_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(month_counts.index, month_counts.values, marker='o', color='black', label='Historical')
        plt.plot(forecast_df['date'], forecast_df['future sightings'], marker='o', color='blue', label='Forecast')
        plt.xlabel("12-Month Period")
        plt.ylabel("Tick Sightings")
        plt.title(f"Tick Sightings Trend (12-Month Intervals) for {location}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


t = TickSightingsAnalyser('Tick Sightings.xlsx')

# #FILTER TESTS
# print("Location Filter Test:")
# print(t.filter_by_location('Manchester'))
# print()

# print("Date Range Filter Test:")
# print(t.filter_by_date_range('2023-01-01', '2023-06-30'))
# print()

# print("Time of Day Filter Test:")
# print(t.filter_by_time_of_day('Morning'))
# print(t.filter_by_time_of_day("06:40"))
# print(t.filter_by_time_of_day("14:00-18:00"))

#USING TWO FILTERS TOGETHER
# print("Location and date range:")
# print(t.filter_by_date_range('2023-01-01', '2023-12-31', t.filter_by_location('Manchester')))
# print("Location and time of day:")
# print(t.filter_by_time_of_day('Night', t.filter_by_location('London')))
# print("Date range and time of day:")
# print(t.filter_by_time_of_day('Afternoon', t.filter_by_date_range('2023-03-01', '2023-09-30')))
print("All three filters:")
df = t.filter_by_date_range('2023-01-01', '2023-12-31', t.filter_by_location('Manchester'))
print(df)

# #GROUPING TESTS
# print("Ticks per location:")
# print(t.show_sighting_per_region())
# print()
# print("Ticks per species:")
# print(t.show_sightings_per_species())
# print()
# print("Ticks per month:")
# print(t.show_sightings_per_month())
# print("Ticks per week:")
# print(t.show_sightings_per_week())

# #AI/ML TESTS
# #FORECASTING TEST
# ts, forecast = t.forecast_tick_activity(months_ahead=24)
# print("Historical monthly counts:")
# print(ts) # ts.head() first 5 rows

# print("\nForecasted values:")
# print(forecast)

# t.plot_forecast(ts, forecast)

# #LOCATION FORECAST TEST
# print("\nLocation Trend Forecast Test:")
# # Call forecast for Manchester, 5 years ahead
# yearly_counts, forecast_df = t.forecast_location_trend("Manchester", years_ahead=10)

# # Print results
# print("Historical yearly counts:")
# print(yearly_counts.reset_index())
# #print(forecast_df)

# print("\nForecasted values:")
# print(forecast_df)
# t.plot_location_forecast("Manchester", yearly_counts, forecast_df)
