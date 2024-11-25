import streamlit as st
import yaml
import os
from run_simulation import main
import json
from datetime import datetime
import pandas as pd
import glob
from src.web.util import parse_log

st.set_page_config(layout="wide")

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'current_day' not in st.session_state:
    st.session_state.current_day = 1
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = {}
if 'history_action' not in st.session_state:
    st.session_state.history_action = {}
if 'history_prompt' not in st.session_state:
    st.session_state.history_prompt = {}
if 'agents' not in st.session_state:
    st.session_state.agents = []

# Load default st.session_state.config
with open('configs/config.yaml', 'r') as file:
    st.session_state.config = yaml.safe_load(file)

st.title("Student Agent Simulation Dashboard")

# Create two columns for the layout
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Configuration")
    
    # Create tabs
    load_tab, run_tab = st.tabs(["Load Existing Log", "Run New Simulation"])
    
    with load_tab:
        # Get list of log directories
        log_dirs = glob.glob('logs/*/')
        if log_dirs:
            selected_log = st.selectbox(
                "Select Log Directory",
                options=log_dirs,
                format_func=lambda x: x.split('/')[-2]
            )

            ### Load and display selected log###
            if st.button("Load Selected Log"):
                logs = [log for log in glob.glob(os.path.join(selected_log, '*.log')) if 'Teacher' not in log]
                
                for log in logs:
                    agent_name = log.split('/')[-1].split('.')[0]
                    st.session_state.agents.append(agent_name)
                    with open(log, 'r') as file:
                        st.session_state.history_prompt[agent_name], st.session_state.history_action[agent_name] = parse_log(file.read())
                st.session_state.simulation_running = True
                st.success(f"Loaded log from {selected_log}")
        else:
            st.warning("No log directories found")
    
    with run_tab:
        # Status value inputs
        st.subheader("Status Values")
        status_values = {
            'loss_mood_study': st.number_input('Loss Mood (Study)', value=st.session_state.config['Status']['loss_mood_study']),
            'loss_energy_study': st.number_input('Loss Energy (Study)', value=st.session_state.config['Status']['loss_energy_study']),
            'loss_mood_take_courses': st.number_input('Loss Mood (Courses)', value=st.session_state.config['Status']['loss_mood_take_courses']),
            'loss_energy_take_courses': st.number_input('Loss Energy (Courses)', value=st.session_state.config['Status']['loss_energy_take_courses']),
            'loss_energy_socialize': st.number_input('Loss Energy (Socialize)', value=st.session_state.config['Status']['loss_energy_socialize']),
            'add_mood_socialize': st.number_input('Add Mood (Socialize)', value=st.session_state.config['Status']['add_mood_socialize']),
            'add_mood_relax': st.number_input('Add Mood (Relax)', value=st.session_state.config['Status']['add_mood_relax']),
            'add_energy_sleep': st.number_input('Add Energy (Sleep)', value=st.session_state.config['Status']['add_energy_sleep']),
            'add_health_exercise': st.number_input('Add Health (Exercise)', value=st.session_state.config['Status']['add_health_exercise']),
        }

        # PDF file upload
        st.subheader("Upload Study Material")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        ### Start real-time simulation ###
        if st.button("Start Real-time Simulation"):
            # Update st.session_state.config with new values
            updated_st.session_state.config = st.session_state.config.copy()
            updated_st.session_state.config['Status'].update(status_values)
            
            if uploaded_file:
                # Save uploaded PDF
                pdf_path = os.path.join('materials', uploaded_file.name)
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                updated_st.session_state.config['System']['PDF_PATH'] = pdf_path

            # Save updated st.session_state.config
            with open('st.session_state.configs/st.session_state.config.yaml', 'w') as file:
                yaml.dump(updated_st.session_state.config, file)

            # Clear previous data
            st.session_state.real_time_data = {}
            st.session_state.simulation_running = True
            st.session_state.current_day = 1
            
            # Start real-time simulation
            placeholder = st.empty()
            for day in range(1, updated_st.session_state.config['System']['DAYS'] + 1):
                # Run simulation for one day
                main(grade=False, real_time=True, current_day=day)
                
                # Load the latest results
                log_dir = updated_st.session_state.config['System']['LOG_PATH'].format(
                    date_time=datetime.now().strftime('%Y%m%d_%H%M%S'))
                with open(f'{log_dir}/agents_history.json', 'r') as file:
                    st.session_state.real_time_data = json.load(file)
                

with col2:
    st.header("Simulation Results")

    if st.session_state.simulation_running:
        # Determine which data to use
        if st.session_state.history_action:
            # Create a row for navigation controls
            nav_col1, nav_col2, nav_col3 = st.columns([1, 10, 1])
            
            # Get max day from actual data
            max_day = st.session_state.config['System']['DAYS'] 
            
            # Initialize selected_day in session state if not exists
            if 'selected_day' not in st.session_state:
                st.session_state.selected_day = max_day
            
            # Left arrow (previous day)
            with nav_col1:
                if st.button("←", key="prev_day"):
                    st.session_state.selected_day = max(1, st.session_state.selected_day - 1)
            
            # Day slider
            with nav_col2:
                st.session_state.selected_day = st.slider(
                    "Select Day", 
                    1, 
                    max_day, 
                    st.session_state.selected_day
                )
            
            # Right arrow (next day)
            with nav_col3:
                if st.button("→", key="next_day"):
                    st.session_state.selected_day = min(max_day, st.session_state.selected_day + 1)

            # Create container for results to maintain state
            results_container = st.container()
            
            with results_container:
                # Display results for selected day
                st.subheader(f"Day {st.session_state.selected_day} Results")

                # Create columns for each agent
                cols = st.columns(len(st.session_state.agents))

                for idx, agent_name in enumerate(st.session_state.agents):
                    with cols[idx]:
                        # Dummy image (replace with actual agent photos later)
                        st.image("https://via.placeholder.com/300", caption=agent_name)
                        
                        # Display actions and final status
                        if st.session_state.selected_day in st.session_state.history_action[agent_name]:
                            actions_data = st.session_state.history_action[agent_name][st.session_state.selected_day]
                            st.write("**Actions:**")
                            for action_data in actions_data:
                                st.write(f"- {action_data['action']}")
                            
                            st.write("**Status:**")
                            final_status = actions_data[-1]['status']  # Get the final status from the last action
                            for status_key, status_value in final_status.items():
                                st.progress(max(0.0, min(1.0, status_value / 100)), 
                                          text=f"{status_key}: {status_value}")
                        else:
                            st.warning(f"No data for day {st.session_state.selected_day}")

        elif st.session_state.real_time_data:
            pass
            