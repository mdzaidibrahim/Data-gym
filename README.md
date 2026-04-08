# Data Cleaning Gym (OpenEnv)

## Problem Description
Data scientists spend nearly 80% of their time cleaning and preparing data. This environment provides a structured reinforcement learning setting where an AI agent can learn to transform dirty DataFrames into clean, standardized datasets.

## Action Space
- `action_type`: "drop_nulls", "fill_nulls", "convert_type", "remove_duplicates", "fix_value"
- `column`: column name to apply the action (optional)
- `value`: value for fill_nulls or convert_type/fix_value operations (optional)

## Observation Space
- `dataframe`: The current Pandas DataFrame serialized as a JSON string (list of dicts).
- `task_description`: Description of the specific data cleaning goal.
- `remaining_steps`: Integer countdown until the episode ends.

## Task Descriptions
- **easy_cleaning**: Removing null values and true duplicates from a simple DataFrame.
- **medium_cleaning**: Standardizing data types (e.g. string to float) and basic formats.
- **hard_cleaning**: Resolving logical inconsistencies like negative salaries and handling complex duplicate logic.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run the environment server: `python app.py` (which will start on `http://0.0.0.0:7860`)
3. Run inference: `python inference.py` (ensure OPENAI_API_KEY or HF_TOKEN is set)
4. Use OpenEnv Validator: `python -m openenv_core validate` (if the exact command fails you can check your system's PATH).

## Baseline Scores
The baseline agent should achieve >0.9 on Easy and Medium, and potentially struggle moderately on Hard until properly trained.
