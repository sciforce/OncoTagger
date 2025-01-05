import pandas as pd
import re
import logging

# Configure logging for tracking extraction issues
logging.basicConfig(
    filename='country_extraction_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths to input and output files
input_file = r"D:\results\filtered_dataset.xlsx"
output_file = r"D:\results\processed_dataset.xlsx"

# Load the dataset (use .head(100) for testing)
df = pd.read_excel(input_file)

# Function to extract the country name directly from the address
def extract_country(address):
    if pd.isna(address):
        return "Not Specified"

    # Check if the last 4 characters are 'USA.'
    if address.strip()[-4:] == "USA.":
        logging.info(f"Extracted country: USA from: {address}")
        return "USA"

    # Match the country name at the end of the address, after a comma
    match = re.search(r",\s*([A-Za-z\s&.'-]+)\s*(?:\d{0,5})?\.*$", address)
    if match:
        country = match.group(1).strip()
        logging.info(f"Extracted country: {country} from: {address}")
        return country
    else:
        logging.info(f"Failed to extract country from: {address}")
        return "Not Specified"

# Apply the extraction function to the dataset
df['Country'] = df['Reprint Addresses'].apply(extract_country)

# Generate summaries of unique countries and their counts
unique_countries = df['Country'].value_counts().reset_index()
unique_countries.columns = ['Country', 'Count']

# Group articles by both 'Country' and 'Publication Year'
country_year_distribution = (
    df.groupby(['Country', 'Publication Year']).size().reset_index(name='Article Count')
)

# Save the results into a new Excel file
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Save the original data with the new Country column
    df.to_excel(writer, sheet_name='Original Data', index=False)
    
    # Save the unique country summary
    unique_countries.to_excel(writer, sheet_name='Unique Countries', index=False)
    
    # Save the country-year distribution
    country_year_distribution.to_excel(writer, sheet_name='Country-Year Distribution', index=False)

print("Script completed successfully.")
