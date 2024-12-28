import numpy as np
import pandas as pd
import random

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 300

# Generating synthetic data for the features
square_footage = np.random.randint(800, 4000, size=num_samples)  # Size of the house (sqft)
num_rooms = np.random.randint(2, 8, size=num_samples)  # Number of rooms
house_age = np.random.randint(0, 50, size=num_samples)  # Age of the house (years)
distance_to_city = np.random.randint(1, 50, size=num_samples)  # Distance to city center (km)
school_rating = np.random.randint(4, 10, size=num_samples)  # Local school rating (0-10)
crime_rate = np.random.uniform(0, 10, size=num_samples)  # Crime rate per 1000 people
median_income = np.random.randint(30, 120, size=num_samples)  # Median income of area ($k)

# Defining a price function based on the features
# Price is influenced by all these factors in a somewhat linear fashion but with some randomness
price = (square_footage * 150) + (num_rooms * 30000) - (house_age * 2000) \
        - (distance_to_city * 5000) + (school_rating * 10000) \
        - (crime_rate * 10000) + (median_income * 5000) + np.random.normal(0, 20000, num_samples)

# Ensuring the price is realistic (non-negative)
price = np.maximum(price, 50000)

# Creating the DataFrame
data = pd.DataFrame({
    'Square Footage (sqft)': square_footage,
    'Number of Rooms': num_rooms,
    'House Age (years)': house_age,
    'Distance to City Center (km)': distance_to_city,
    'School Rating': school_rating,
    'Crime Rate (incidents per 1000)': crime_rate,
    'Median Income ($k)': median_income,
    'Price ($)': price
})

# Saving to CSV
data.to_csv('regression_data.csv', index=False)
print("Data saved to 'regression_data.csv'")
