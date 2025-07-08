import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine
from sqlalchemy import text 
from sqlalchemy.engine import URL 


connection_string="DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.10.67.9,4701;DATABASE=AIML ;UID=mis_aiml;PWD=Aiml@96985"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
engine = create_engine(connection_url)

def get_data_from_db(query):
    # Update connection string with the provided credentials
    #engine = create_engine('mssql://MDM_VIEW:EsyaSoft@321@host/AIML')
    connection_string="DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.10.67.9,4701;DATABASE=AIML ;UID=mis_aiml;PWD=Aiml@96985"
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    engine = create_engine(connection_url)
    data = pd.read_sql(query,engine)
    return data


def get_data ():
    q = '''    
    select SubDiv , Meters  
    , Active_Installers  
    ,  Meters_Installed_Already 
    , total_meter_installed_till_date ,
      Subdivision_Productivity
        , Open_Date from  t_subdiv_data_scheduler
    '''
    data = get_data_from_db(q)
    return data

st.set_page_config(
    page_title="Catch Up Plan Predictor",
    layout="wide"
)



input_df = get_data()
    
    # Fix data types
numeric_cols = ['Meters', 'total_meter_installed_till_date', 'Subdivision_Productivity', 'Active_Installers']
for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
input_df['Open_Date'] = pd.to_datetime(input_df['Open_Date'], errors='coerce')
# input_df['date'] = pd.to_datetime(input_df['date'], errors='coerce')
input_df = input_df.dropna(subset=numeric_cols)




input_df['remaining_meter'] = input_df['Meters'] - input_df['total_meter_installed_till_date']
input_df['required_time'] = input_df['remaining_meter'] // (input_df['Active_Installers'] * input_df['Subdivision_Productivity'])
input_df['required_time'] = input_df['required_time'].astype('int32')
input_df['Open_Date'] = pd.to_datetime(input_df['Open_Date'])
input_df['required_time'] = input_df['required_time'].apply(

    lambda x: pd.Timestamp.today() + pd.to_timedelta(x, unit='D')

)
 
input_df['required_time'] = input_df['required_time'].dt.strftime('%d-%m-%Y')

# Save original DataFrame
original_df = input_df.copy()

# Sidebar for input modifications
st.sidebar.header("Modify Parameters")

update_mode = st.sidebar.radio("Update Mode", ["Single Subdivision", "All Subdivisions"])

# Initialize updated DataFrame
updated_df = input_df.copy()

if update_mode == "Single Subdivision":
    selected_subdiv = st.sidebar.selectbox("Select Subdivision", input_df['SubDiv'])
    idx = input_df[input_df['SubDiv'] == selected_subdiv].index[0]
    active_installers = st.sidebar.number_input(
        "Active Installers",
        min_value=1.0,
        value=input_df.loc[idx, 'Active_Installers'],
        step=1.0
    )
    subdivision_productivity = st.sidebar.number_input(
        "Subdivision Productivity",
        min_value=1.0,
        value=input_df.loc[idx, 'Subdivision_Productivity'],
        step=1.0
    )
    # Update single subdivision
    updated_df.loc[idx, 'Active_Installers'] = active_installers
    updated_df.loc[idx, 'Subdivision_Productivity'] = subdivision_productivity
else:
    global_active_installers = st.sidebar.number_input(
        "Global Active Installers",
        min_value=1.0,
        #value=updated_df['Active_Installers'].mean(),
        value=10,
        step=1.0
    )
    global_subdivision_productivity = st.sidebar.number_input(
        "Global Subdivision Productivity",
        min_value=1.0,
        # value=updated_df['Subdivision_Productivity'].mean(),
        value=7,
        step=1.0
    )
    # Update all subdivisions
    updated_df['Active_Installers'] = global_active_installers
    updated_df['Subdivision_Productivity'] = global_subdivision_productivity

# Recalculate required_time for updated DataFrame
updated_df['remaining_meter'] = updated_df['Meters'] - updated_df['total_meter_installed_till_date']
updated_df['required_time'] = updated_df['remaining_meter'] // (updated_df['Active_Installers'] * updated_df['Subdivision_Productivity'])
updated_df['required_time'] = updated_df['required_time'].astype('int32')
updated_df['Open_Date'] = pd.to_datetime(updated_df['Open_Date'])
# updated_df['required_time'] = updated_df.apply(
#     lambda row: row['Open_Date'] + pd.to_timedelta(row['required_time'], unit='D'),
#     axis=1
# )
updated_df['required_time'] = updated_df['required_time'].apply(
   lambda x: pd.Timestamp.today() + pd.to_timedelta(x, unit='D')
)
updated_df['required_time'] = updated_df['required_time'].dt.strftime('%d-%m-%Y')

# Calculate per day and monthly installation rates
original_df['per_day_installation'] = original_df['Active_Installers'] * original_df['Subdivision_Productivity']
updated_df['per_day_installation'] = updated_df['Active_Installers'] * updated_df['Subdivision_Productivity']
original_df['monthly_installation'] = original_df['per_day_installation'] * 30
updated_df['monthly_installation'] = updated_df['per_day_installation'] * 30

# Calculate days saved
original_df['required_time_dt'] = pd.to_datetime(original_df['required_time'], format='%d-%m-%Y')
updated_df['required_time_dt'] = pd.to_datetime(updated_df['required_time'], format='%d-%m-%Y')
days_saved = (original_df['required_time_dt'] - updated_df['required_time_dt']).dt.days
days_saved_df = pd.DataFrame({
    'SubDiv': original_df['SubDiv'],
    'Days_Saved': days_saved
})

# End Date Calculator
st.sidebar.header("End Date Calculator")
calc_mode = st.sidebar.radio("Calculate for", ["Single Subdivision", "All Subdivisions"])
desired_end_date = st.sidebar.date_input("Desired End Date", value=datetime(2026, 1, 31))
fix_parameter = st.sidebar.selectbox("Fix Parameter", ["Active Installers", "Subdivision Productivity"])


# Calculate required parameters
required_params_df = updated_df.copy()
if calc_mode == "Single Subdivision":
    calc_subdiv = st.sidebar.selectbox("Select Subdivision for Calculation", required_params_df['SubDiv'])
    calc_idx = required_params_df[required_params_df['SubDiv'] == calc_subdiv].index[0]
    # Convert desired_end_date to Timestamp for compatibility
    desired_end_date_ts = pd.Timestamp(desired_end_date)
    days_required = (desired_end_date_ts - required_params_df.loc[calc_idx, 'Open_Date']).days
    if days_required <= 0:
        st.sidebar.error("Desired end date must be after the Open Date.")
        required_params_df.loc[calc_idx, 'Active_Installers'] = float('inf')
        required_params_df.loc[calc_idx, 'Subdivision_Productivity'] = float('inf')
    else:
        remaining_meters = required_params_df.loc[calc_idx, 'remaining_meter']
        required_product = remaining_meters / days_required if days_required > 0 else float('inf')
        if fix_parameter == "Active Installers":
            current_installers = required_params_df.loc[calc_idx, 'Active_Installers']
            required_params_df.loc[calc_idx, 'Subdivision_Productivity'] = required_product / current_installers if current_installers > 0 else float('inf')
            required_params_df.loc[calc_idx, 'Active_Installers'] = current_installers
        else:
            current_productivity = required_params_df.loc[calc_idx, 'Subdivision_Productivity']
            required_params_df.loc[calc_idx, 'Active_Installers'] = required_product / current_productivity if current_productivity > 0 else float('inf')
            required_params_df.loc[calc_idx, 'Subdivision_Productivity'] = current_productivity
else:
    # Convert desired_end_date to Timestamp for compatibility
    desired_end_date_ts = pd.Timestamp(desired_end_date)

    for idx in required_params_df.index:
        days_required = (desired_end_date_ts - required_params_df.loc[idx, 'Open_Date']).days
        if days_required <= 0:
            required_params_df.loc[idx, 'Active_Installers'] = float('inf')
            required_params_df.loc[idx, 'Subdivision_Productivity'] = float('inf')
        else:
            remaining_meters = required_params_df.loc[idx, 'remaining_meter']
            required_product = remaining_meters / days_required if days_required > 0 else float('inf')
            if fix_parameter == "Active Installers":
                current_installers = required_params_df.loc[idx, 'Active_Installers']
                required_params_df.loc[idx, 'Subdivision_Productivity'] = required_product / current_installers if current_installers > 0 else float('inf')
                required_params_df.loc[idx, 'Active_Installers'] = current_installers
            else:
                current_productivity = required_params_df.loc[idx, 'Subdivision_Productivity']
                required_params_df.loc[idx, 'Active_Installers'] = required_product / current_productivity if current_productivity > 0 else float('inf')
                required_params_df.loc[idx, 'Subdivision_Productivity'] = current_productivity
        
st.sidebar.markdown(
    "<p style='font-size: 14px;'>🔧 Developed by <em>Esyasoft Analytics Department</em></p>",
    unsafe_allow_html=True
)
# Main content with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Tables", "Completion Dates", "Per Day Installation", "Monthly Installation", "Days Saved"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Data")
        st.dataframe(original_df[['SubDiv','Meters','total_meter_installed_till_date','Active_Installers', 'Subdivision_Productivity', 'required_time', 'per_day_installation', 'monthly_installation']], use_container_width=True)
    with col2:
        st.subheader("Updated Data")
        st.dataframe(updated_df[['SubDiv','Meters', 'total_meter_installed_till_date','Active_Installers', 'Subdivision_Productivity', 'required_time', 'per_day_installation', 'monthly_installation']], use_container_width=True)
    st.subheader("Days Saved")
    st.dataframe(days_saved_df, use_container_width=True)
    st.subheader("Required Parameters for Desired End Date")
    required_params_df['Active_Installers']=required_params_df['Active_Installers'].round(0).astype('int32')

    required_params_df['Subdivision_Productivity']=required_params_df['Subdivision_Productivity'].round(0).astype('int32')
    st.dataframe(required_params_df[['SubDiv', 'Active_Installers', 'Subdivision_Productivity']], use_container_width=True)

# Create comparison plot for completion dates
with tab2:
    fig_completion = go.Figure()
    fig_completion.add_trace(
        go.Scatter(
            x=original_df['SubDiv'],
            y=original_df['required_time_dt'],
            mode='lines+markers',
            name='Original Completion Date',
            line=dict(color='blue'),
            marker=dict(size=8)
        )
    )
    fig_completion.add_trace(
        go.Scatter(
            x=updated_df['SubDiv'],
            y=updated_df['required_time_dt'],
            mode='lines+markers',
            name='Updated Completion Date',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8)
        )
    )
    fig_completion.update_layout(
        title='Comparison of Completion Dates',
        xaxis_title='Subdivision',
        yaxis_title='Completion Date',
        xaxis=dict(tickangle=45),
        yaxis=dict(tickformat='%d-%m-%Y'),
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig_completion, use_container_width=True)

# Create comparison plot for per day installation
with tab3:
    fig_per_day = go.Figure()
    fig_per_day.add_trace(
        go.Bar(
            x=original_df['SubDiv'],
            y=original_df['per_day_installation'],
            name='Original Per Day Installation',
            marker_color='blue'
        )
    )
    fig_per_day.add_trace(
        go.Bar(
            x=updated_df['SubDiv'],
            y=updated_df['per_day_installation'],
            name='Updated Per Day Installation',
            marker_color='red'
        )
    )
    fig_per_day.update_layout(
        title='Per Day Meter Installation Comparison',
        xaxis_title='Subdivision',
        yaxis_title='Meters Installed Per Day',
        xaxis=dict(tickangle=45),
        barmode='group',
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig_per_day, use_container_width=True)

# Create comparison plot for monthly installation
with tab4:
    fig_monthly = go.Figure()
    fig_monthly.add_trace(
        go.Bar(
            x=original_df['SubDiv'],
            y=original_df['monthly_installation'],
            name='Original Monthly Installation',
            marker_color='blue'
        )
    )
    fig_monthly.add_trace(
        go.Bar(
            x=updated_df['SubDiv'],
            y=updated_df['monthly_installation'],
            name='Updated Monthly Installation',
            marker_color='red'
        )
    )
    fig_monthly.update_layout(
        title='Monthly Meter Installation Comparison',
        xaxis_title='Subdivision',
        yaxis_title='Meters Installed Per Month',
        xaxis=dict(tickangle=45),
        barmode='group',
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

# Create bar plot for days saved
with tab5:
    fig_days_saved = go.Figure()
    fig_days_saved.add_trace(
        go.Bar(
            x=days_saved_df['SubDiv'],
            y=days_saved_df['Days_Saved'],
            name='Days Saved',
            marker_color='green'
        )
    )
    fig_days_saved.update_layout(
        title='Days Saved by Increased Productivity',
        xaxis_title='Subdivision',
        yaxis_title='Days Saved',
        xaxis=dict(tickangle=45),
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig_days_saved, use_container_width=True)