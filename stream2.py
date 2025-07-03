# import streamlit as st
# import pandas as pd
# from io import BytesIO
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import webbrowser
# st.set_page_config(
#     page_title="Catch Up Plan Predictor",
#     layout="wide"  # Options: "centered" (default), "wide"
# )

# def helper(df):
#     total_target_meters =df['monthly_mi_plan'].sum()-8062-2500-131385-9540
#     cutoff_date = datetime(2025, 6, 30)  # CHANGE THIS VALUE as needed

#     # Clean and format data
#     df['Month'] = pd.to_datetime(df['Month'], format='%b %y', errors='coerce')
#     df = df.dropna(subset=['Month'])
#     df['Actual MI Done'] = df['Actual MI Done'].fillna(0)
#     df['monthly_mi_plan'] = pd.to_numeric(df['monthly_mi_plan'], errors='coerce').fillna(0)
#     df['team_avail_1ph_3ph_wc '] = pd.to_numeric(df['team_avail_1ph_3ph_wc '], errors='coerce').fillna(0).astype(int)
#     df['installed_1ph'] = pd.to_numeric(df['installed_1ph'], errors='coerce').fillna(0)
#     df['installed_3ph_wc'] = pd.to_numeric(df['installed_3ph_wc'], errors='coerce').fillna(0)

#     # Filter historical data and generate future months
#     historical_df = df[df['Month'] <= cutoff_date]
#     future_start_date = cutoff_date + relativedelta(months=1)
#     future_start_date = future_start_date.replace(day=1)

#     future_months = []
#     current_date = future_start_date
#     end_date = datetime(2026, 6, 30)
#     while current_date <= end_date:
#         future_months.append(current_date)
#         current_date += relativedelta(months=1)
#     future_months = [d.strftime('%b %Y') for d in future_months]

#     # Modified analyze_meter_installation function
#     def analyze_meter_installation(df, total_target, cutoff_date, future_months):
#         result = []
#         not_dic = {}
#         un_temp = 0  # Backlog

#         # Process historical data
#         historical_df = df[df['Month'] <= cutoff_date]
#         total_installed = historical_df['Actual MI Done'].sum()
#         remaining_meters = total_target - total_installed

#         for index, row in historical_df.iterrows():
#             one_ph_count = pd.to_numeric(row.get('installed_1ph', 0), errors='coerce')
#             three_ph_count = pd.to_numeric(row.get('installed_3ph_wc', 0), errors='coerce')
#             actual_done = pd.to_numeric(row.get('Actual MI Done', 0), errors='coerce')
#             team_available = pd.to_numeric(row.get('team_avail_1ph_3ph_wc ', 0), errors='coerce')
            

#             one_ph_count = 0 if pd.isna(one_ph_count) else one_ph_count
#             three_ph_count = 0 if pd.isna(three_ph_count) else three_ph_count
#             actual_done = 0 if pd.isna(actual_done) else actual_done
#             team_available = 0 if pd.isna(team_available) else team_available

#             total_meter = one_ph_count + three_ph_count + un_temp
#             total_capacity = int(team_available * 25 * 12)
#             uninstalled = total_meter - actual_done

#             if uninstalled > 0 and total_capacity > total_meter:
#                 not_dic[row['Month'].strftime('%b %Y')] = (
#                     f"They had capacity of {total_capacity:,} but did not utilize it, "
#                     f"{uninstalled:,} meter(s) left in {row['Month'].strftime('%b %Y')}."
#                 )
#                 result.append({
#                     'Month': row['Month'].strftime('%b %Y'),
#                     'monthly_mi_plan': row['monthly_mi_plan'],
#                     'Actual MI Done': actual_done,
#                     'team_avail_1ph_3ph_wc ': team_available,
#                     'available_capacity': total_capacity,
#                     'planned_meter_1ph_3ph': total_meter,
#                     'revised_plan_meters': total_meter,
#                     'uninstalled': uninstalled,
#                     'capacity_utilized': False,
#                     'capacity_shortfall': 0
#                 })
#                 un_temp = uninstalled
#             elif total_capacity < total_meter:
#                 shortage = total_meter - total_capacity
#                 result.append({
#                     'Month': row['Month'].strftime('%b %Y'),
#                     'monthly_mi_plan': row['monthly_mi_plan'],
#                     'Actual MI Done': actual_done,
#                     'team_avail_1ph_3ph_wc ': team_available,
#                     'available_capacity': total_capacity,
#                     'planned_meter_1ph_3ph': total_meter,
#                     'revised_plan_meters': min(total_meter, total_capacity),
#                     'uninstalled': uninstalled,
#                     'capacity_shortfall': shortage,
#                     'capacity_utilized': True
#                 })
#                 un_temp = uninstalled if uninstalled > 0 else 0
#             else:
#                 result.append({
#                     'Month': row['Month'].strftime('%b %Y'),
#                     'monthly_mi_plan': row['monthly_mi_plan'],
#                     'Actual MI Done': actual_done,
#                     'team_avail_1ph_3ph_wc ': team_available,
#                     'available_capacity': total_capacity,
#                     'planned_meter_1ph_3ph': total_meter,
#                     'revised_plan_meters': total_meter,
#                     'uninstalled': 0,
#                     'capacity_utilized': True,
#                     'capacity_shortfall': 0
#                 })
#                 un_temp = 0

#         # Create future plan using provided team numbers
#         monthly_target = remaining_meters // len(future_months) if future_months else 0
#         remaining = remaining_meters + un_temp
#         future_df = df[df['Month'] > cutoff_date]

#         for month in future_months:
#             # Get team availability for the month from the dataset
#             team_available = future_df[future_df['Month'].dt.strftime('%b %Y') == month]['team_avail_1ph_3ph_wc ']
#             team_available = team_available.iloc[0] if not team_available.empty else 0
#             monthly_capacity = int(team_available * 25 * 12) if team_available > 0 else 0

#             planned_meters = min(monthly_target + un_temp, monthly_capacity, remaining)
#             installed = min(planned_meters, monthly_capacity)
#             uninstalled = planned_meters - installed
#             shortfall = planned_meters - monthly_capacity if planned_meters > monthly_capacity else 0

#             result.append({
#                 'Month': month,
#                 'monthly_mi_plan': planned_meters,
#                 'Actual MI Done': 0,
#                 'team_avail_1ph_3ph_wc ': team_available,
#                 'available_capacity': monthly_capacity,
#                 'planned_meter_1ph_3ph': planned_meters,
#                 'revised_plan_meters': installed,
#                 'uninstalled': uninstalled,
#                 'capacity_shortfall': shortfall,
#                 'capacity_utilized': installed == monthly_capacity
#             })

#             un_temp = uninstalled if uninstalled > 0 else 0
#             remaining -= installed

#         result_df = pd.DataFrame(result)
#         return result_df, not_dic, total_installed, remaining_meters

#     # Run the analysis
#     result_df, not_dic, total_installed, remaining_meters = analyze_meter_installation(df, total_target_meters, cutoff_date, future_months)
    

#     # Print summary
#     print("Meter Installation Plan Summary")
#     print(f"Total Target: {total_target_meters:,} meters")
#     print(f"Installed (Oct 2024 - {cutoff_date.strftime('%b %Y')}): {total_installed:,.0f} meters")
#     print(f"Remaining ({future_months[0]} - Jun 2026): {remaining_meters:,.0f} meters")
#     print("Interesting Fact: In April 2025, teams had a capacity of 24,546 meters but planned only 24,431 meters, indicating underutilized capacity.")
#     print("\nDetailed plan saved to 'meter_installation_plan.csv'")

#     # Visualize historical performance
#     # Visualize historical performance
#     historical_df = result_df[result_df['Month'].isin([m.strftime('%b %Y') for m in df[df['Month'] <= cutoff_date]['Month']])]
#     plt.figure(figsize=(10, 6))
#     plt.bar(historical_df['Month'], historical_df['monthly_mi_plan'], width=0.35, label='Planned', color='skyblue')
#     plt.bar(historical_df['Month'], historical_df['Actual MI Done'], width=0.35, label='Actual', color='limegreen', alpha=0.6)
#     plt.xlabel('Month')
#     plt.ylabel('Meters')
#     plt.title(f'Historical Performance (Oct 2024 - {cutoff_date.strftime("%b %Y")})')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     st.subheader("📊 Historical Performance (Oct 2024 - Jun 2025)")
#     st.pyplot(plt.gcf())  # 👈 This shows it in Streamlit
#     plt.close()


#     # Visualize future plan
#     future_df = result_df[result_df['Month'].isin(future_months)]
#     plt.figure(figsize=(10, 6))
#     plt.plot(future_df['Month'], future_df['revised_plan_meters'], marker='o', label='Planned Meters', color='skyblue')
#     plt.plot(future_df['Month'], future_df['available_capacity'], marker='s', label='Team Capacity', color='limegreen')
#     plt.xlabel('Month')
#     plt.ylabel('Meters')
#     plt.title(f'Future Plan ({future_months[0]} - Jun 2026)')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     st.subheader("📈 Future Plan (Jul 2025 - Jun 2026)")
#     st.pyplot(plt.gcf())  # 👈 This shows it in Streamlit
#     plt.close()


#     # Print underutilized capacity insights
#     print("\nUnderutilized Capacity:")
#     for month, message in not_dic.items():
#         print(message)
#     return result_df, not_dic, total_installed, remaining_meters



# def to_excel_download(df: pd.DataFrame) -> BytesIO:
#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name='Modified')
#     output.seek(0)
#     return output


# st.title("CATCHUP PLAN PREDICTOR ")


# uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
# col1, col2 = st.columns([1, 2]) 
# if uploaded_file:
    
#     original_df = pd.read_excel(uploaded_file)

#     st.subheader("Original Excel Data")
#     st.dataframe(original_df)

#     modified_df, not_dic, total_installed, remaining_meters = helper(original_df)


#     st.subheader("COMPUTING PREDICTION")
    
#     st.dataframe(modified_df)

#     excel_file = to_excel_download(modified_df)

#     st.download_button(
#         label="📥 Download Predicted Result",
#         data=excel_file,
#         file_name="modified_excel.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )
#     if st.button('Click to See Analysis'):
#         webbrowser.open_new_tab('https://openai.com')


import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import webbrowser
import plotly.graph_objects as go
import plotly.express as px
# Page config
st.set_page_config(
    page_title="Catch Up Plan Predictor",
    layout="wide"
)

# Sidebar for upload
with st.sidebar:
    st.title("📂 Upload File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    st.markdown("---")
    st.markdown("🔧 Developed by **AMAN**")

# Function to handle Excel export
def to_excel_download(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Modified')
    output.seek(0)
    return output

# Core helper logic
def helper(df,proc):
    import uuid  # Add at the top if not already imported
    unique_id = str(uuid.uuid4())
    total_target_meters = df['monthly_mi_plan'].sum() - 8062 - 2500 - 131385 - 9540
    cutoff_date = datetime(2025, 6, 30)

    df['Month'] = pd.to_datetime(df['Month'], format='%b %y', errors='coerce')
    df = df.dropna(subset=['Month']).copy()
    df['Actual MI Done'] = df['Actual MI Done'].fillna(0)
    df['monthly_mi_plan'] = pd.to_numeric(df['monthly_mi_plan'], errors='coerce').fillna(0)
    df['team_avail_1ph_3ph_wc '] = pd.to_numeric(df['team_avail_1ph_3ph_wc '], errors='coerce').fillna(0).astype(int)
    df['installed_1ph'] = pd.to_numeric(df['installed_1ph'], errors='coerce').fillna(0)
    df['installed_3ph_wc'] = pd.to_numeric(df['installed_3ph_wc'], errors='coerce').fillna(0)

    historical_df = df[df['Month'] <= cutoff_date]
    future_months = [(cutoff_date + relativedelta(months=+i)).replace(day=1) for i in range(1, 13)]
    future_months_str = [d.strftime('%b %Y') for d in future_months]

    def analyze_meter_installation(df, total_target, cutoff_date, future_months):
        result = []
        not_dic = {}
        un_temp = 0
        historical_df = df[df['Month'] <= cutoff_date]
        total_installed = historical_df['Actual MI Done'].sum()
        remaining_meters = total_target - total_installed

        for index, row in historical_df.iterrows():
            one_ph = pd.to_numeric(row.get('installed_1ph', 0), errors='coerce')
            three_ph = pd.to_numeric(row.get('installed_3ph_wc', 0), errors='coerce')
            actual_done = pd.to_numeric(row.get('Actual MI Done', 0), errors='coerce')
            team_avail = pd.to_numeric(row.get('team_avail_1ph_3ph_wc ', 0), errors='coerce')

            one_ph = 0 if pd.isna(one_ph) else one_ph
            three_ph = 0 if pd.isna(three_ph) else three_ph
            actual_done = 0 if pd.isna(actual_done) else actual_done
            team_avail = 0 if pd.isna(team_avail) else team_avail

            total_meter = one_ph + three_ph + un_temp
            total_capacity = int(team_avail * 25 * proc)
            uninstalled = total_meter - actual_done

            if uninstalled > 0 and total_capacity > total_meter:
                not_dic[row['Month'].strftime('%b %Y')] = (
                    f"They had capacity of {total_capacity:,} but did not utilize it, "
                    f"{uninstalled:,} meter(s) left in {row['Month'].strftime('%b %Y')}."
                )
                revised_plan = total_meter
                un_temp = uninstalled
            elif total_capacity < total_meter:
                revised_plan = min(total_meter, total_capacity)
                un_temp = uninstalled
            else:
                revised_plan = total_meter
                un_temp = 0

            result.append({
                'Month': row['Month'].strftime('%b %Y'),
                'monthly_mi_plan': row['monthly_mi_plan'],
                'Actual MI Done': actual_done,
                'team_avail_1ph_3ph_wc ': team_avail,
                'available_capacity': total_capacity,
                'planned_meter_1ph_3ph': total_meter,
                'revised_plan_meters': revised_plan,
                'uninstalled': uninstalled,
                'capacity_shortfall': max(0, total_meter - total_capacity),
                'capacity_utilized': revised_plan >= total_capacity
            })
     
            

        monthly_target = remaining_meters // len(future_months) if future_months else 0
        remaining = remaining_meters + un_temp
        future_df = df[df['Month'] > cutoff_date]

        for month in future_months:
            team_avail = future_df[future_df['Month'].dt.strftime('%b %Y') == month]['team_avail_1ph_3ph_wc ']

            team_avail = team_avail.iloc[0] if not team_avail.empty else 0
            monthly_capacity = int(team_avail * 25 * proc)

            planned = min(monthly_target + un_temp, monthly_capacity, remaining)
            installed = min(planned, monthly_capacity)
            uninstalled = planned - installed
            shortfall = planned - monthly_capacity if planned > monthly_capacity else 0

            result.append({
                'Month': month,
                'revised_monthly_mi_plan': planned,
                'Actual MI Done': 0,
                'team_avail_1ph_3ph_wc ': team_avail,
                'available_capacity': monthly_capacity,
                'planned_meter_1ph_3ph': planned,
                'revised_plan_meters': installed,
                'uninstalled': uninstalled,
                'capacity_shortfall': shortfall,
                'capacity_utilized': installed == monthly_capacity
            })

            un_temp = uninstalled if uninstalled > 0 else 0
            remaining -= installed

        return pd.DataFrame(result), not_dic, total_installed, remaining_meters

    
    result_df, not_dic, total_installed, remaining_meters = analyze_meter_installation(df, total_target_meters, cutoff_date, future_months_str)

    
    col1, col2 = st.columns(2)

    temp_df=result_df.pop('revised_monthly_mi_plan')
    result_df.insert(2, 'revised_monthly_mi_plan',temp_df)
    with col1:
        st.markdown("##### 📊 Historical Performance (Oct 2024 - Jun 2025)")

        hist_df = result_df[result_df['Month'].isin([m.strftime('%b %Y') for m in df[df['Month'] <= cutoff_date]['Month']])]

        # Create grouped bar chart using plotly.graph_objects for detailed control
        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            x=hist_df['Month'],
            y=hist_df['monthly_mi_plan'],
            name='Planned',
            marker_color='skyblue',
            hovertemplate='Month: %{x}<br>Planned: %{y}<extra></extra>'
        ))

        fig1.add_trace(go.Bar(
            x=hist_df['Month'],
            y=hist_df['Actual MI Done'],
            name='Actual',
            marker_color='limegreen',
            opacity=0.7,
            hovertemplate='Month: %{x}<br>Actual: %{y}<extra></extra>'
        ))

        fig1.update_layout(
            plot_bgcolor='white',   # Background inside the plotting area
            #paper_bgcolor='white',
            barmode='group',
            title='Planned vs Actual',
            xaxis_title='Month',
            yaxis_title='Meters',
            legend_title_text='Legend',
            template='plotly_white',
            xaxis_tickangle=-45,
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=100)
        )

        st.plotly_chart(fig1, use_container_width=True,key=f"historical_plot_{unique_id}")


# --- Plot 2: Future Plan ---

    with col2:
        st.markdown("##### 📈 Future Plan (Jul 2025 - Jun 2026)")

        fut_df = result_df[result_df['Month'].isin(future_months_str)]

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=fut_df['Month'],
            y=fut_df['revised_plan_meters'],
            mode='lines+markers',
            name='Planned Meters',
            line=dict(color='dodgerblue', width=3),
            marker=dict(size=8),
            hovertemplate='Month: %{x}<br>Planned Meters: %{y}<extra></extra>'
        ))

        fig2.add_trace(go.Scatter(
            x=fut_df['Month'],
            y=fut_df['available_capacity'],
            mode='lines+markers',
            name='Team Capacity',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=8, symbol='square'),
            hovertemplate='Month: %{x}<br>Team Capacity: %{y}<extra></extra>'
        ))

        fig2.update_layout(
            plot_bgcolor='white',   # Background inside the plotting area
            #paper_bgcolor='white',
            title='Future Plan vs Capacity',
            xaxis_title='Month',
            yaxis_title='Meters',
            legend_title_text='Legend',
            template='plotly_white',
            xaxis_tickangle=-45,
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=100)
        )

        st.plotly_chart(fig2, use_container_width=True,key=f"future_plot_{unique_id}")


    return result_df, not_dic, total_installed, remaining_meters

# Main app logic
st.title("📡 Catch Up Plan Predictor")

if uploaded_file:
    original_df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Excel Data")
    with st.expander("🔍 View Raw Data"):
        st.dataframe(original_df, use_container_width=True)
    
    proc = st.number_input("Enter productivity (default is 12)", min_value=1, max_value=30, value=12)

    try:
        proc_value = int(proc)
    except ValueError:
        st.error("Please enter a valid number.")
        proc_value = None
    if proc_value is not None:
        modified_df, not_dic, total_installed, remaining_meters = helper(original_df, proc_value)

    #modified_df, not_dic, total_installed, remaining_meters = helper(original_df,proc_value)

    total_target_meters = original_df['monthly_mi_plan'].sum()-8602 - 2500 - 131385 - 9540

    st.subheader("📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Total Target", f"{int(total_target_meters):,} meters")
    col2.metric("✅ Installed Till Jun 2025", f"{int(total_installed):,} meters")
    col3.metric("📉 Remaining", f"{int(remaining_meters):,} meters")

    st.subheader("📈 Predicted Meter Installation Plan")
    with st.expander("📋 View Computed Dataframe"):
        st.dataframe(modified_df, use_container_width=True)

    st.download_button(
        label="📥 Download Predicted Result (Excel)",
        data=to_excel_download(modified_df),
        file_name="modified_excel.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if not_dic:
        st.subheader("⚠️ Underutilized Capacity Insights")
        for message in not_dic.values():
            st.markdown(f"- {message}")

    st.markdown("---")
    if st.button('🌐 Click to Visit Analysis Reference'):
        webbrowser.open_new_tab('https://openai.com')

