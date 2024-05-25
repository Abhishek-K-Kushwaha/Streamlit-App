import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

# Function to convert time string to datetime object
def time_to_datetime(time_str):
    return datetime.strptime(time_str, '%I:%M:%S %p')

def process_data(input_df):
    # Prepare the output dataframe with the required columns
    output_df = pd.DataFrame(columns=['date', 'pick_activities', 'place_activities', 'inside_duration', 'outside_duration'])

    # Group data by date
    grouped = input_df.groupby('date')

    # Process each group
    for date, group in grouped:
        # Initialize counts and durations
        pick_count = 0
        place_count = 0
        inside_duration = timedelta()
        outside_duration = timedelta()
        
        # Sort the group by time to calculate durations
        group = group.sort_values(by='time')
        
        # Variables to keep track of the last inside and outside times
        last_inside_time = None
        last_outside_time = None
        
        for i, row in group.iterrows():
            # Count pick and place activities
            if row['activity'] == 'picked':
                pick_count += 1
            elif row['activity'] == 'placed':
                place_count += 1
            
            # Calculate durations
            current_time = time_to_datetime(row['time'])
            
            if row['position'].strip().lower() == 'inside':
                if last_inside_time is not None:
                    inside_duration += current_time - last_inside_time
                last_inside_time = current_time
            else:
                if last_outside_time is not None:
                    outside_duration += current_time - last_outside_time
                last_outside_time = current_time
        
        # Append results to the output dataframe
        new_row = pd.DataFrame({
            'date': [date],
            'pick_activities': [pick_count],
            'place_activities': [place_count],
            'inside_duration': [inside_duration],
            'outside_duration': [outside_duration]
        })
        
        output_df = pd.concat([output_df, new_row], ignore_index=True)
    
    return output_df

# Streamlit app
st.title("Activity Data Processor")

# URL of the raw CSV file on GitHub
csv_url = "raw_data.csv"

# Read the CSV file from the URL
input_df = pd.read_csv(csv_url)
output_df = process_data(input_df)

st.write("Processed Data:")
st.dataframe(output_df)

# Provide an option to download the processed data
output_file = 'rawdata_output.xlsx'
with pd.ExcelWriter(output_file) as writer:
    output_df.to_excel(writer, sheet_name='output', index=False)

with open(output_file, 'rb') as f:
    st.download_button('Download Processed Data as Excel', f, file_name=output_file)
