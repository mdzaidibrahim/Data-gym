import pandas as pd
import numpy as np
from env.models import Observation, Action
from env.tasks import get_task
from env.grader import grade

class DataCleaningEnv:
    def __init__(self, task_name: str, max_steps: int = 10):
        self.task_name = task_name
        self.max_steps = max_steps
        self.start_df, self.expected_df, self.task_description = get_task(task_name)
        self.current_df = self.start_df.copy()
        self.steps_taken = 0
        self.last_score = grade(self.current_df, self.expected_df)

    def reset(self) -> Observation:
        self.current_df = self.start_df.copy()
        self.steps_taken = 0
        self.last_score = grade(self.current_df, self.expected_df)
        return self.state()

    def state(self) -> Observation:
        view_df = self.current_df.copy()
        try:
            view_df = view_df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
            view_df = view_df.where(pd.notnull(view_df), None)
            df_dict = view_df.to_dict(orient="records")
        except:
            df_dict = []

        return Observation(
            dataframe=df_dict,
            task_description=self.task_description,
            remaining_steps=self.max_steps - self.steps_taken
        )

    def step(self, action: Action):
        if self.steps_taken >= self.max_steps:
            return self.state(), 0.0, True, {"error": "Max steps reached"}

        error = None
        try:
            if action.action_type == "drop_nulls":
                if action.column:
                    self.current_df = self.current_df.dropna(subset=[action.column]).reset_index(drop=True)
                else:
                    self.current_df = self.current_df.dropna().reset_index(drop=True)
            elif action.action_type == "fill_nulls":
                if action.column and action.value is not None:
                    self.current_df[action.column] = self.current_df[action.column].fillna(action.value)
                else:
                    error = "column and value are required for fill_nulls"
            elif action.action_type == "convert_type":
                if action.column and action.value:
                    if action.value == "int":
                        self.current_df[action.column] = pd.to_numeric(self.current_df[action.column], errors='coerce').astype('Int64')
                    elif action.value == "float":
                        self.current_df[action.column] = pd.to_numeric(self.current_df[action.column], errors='coerce').astype(float)
                    elif action.value == "str":
                        self.current_df[action.column] = self.current_df[action.column].astype(str)
                else:
                    error = "column and value (type) are required for convert_type"
            elif action.action_type == "remove_duplicates":
                if action.column:
                    self.current_df = self.current_df.drop_duplicates(subset=[action.column]).reset_index(drop=True)
                else:
                    self.current_df = self.current_df.drop_duplicates().reset_index(drop=True)
            elif action.action_type == "fix_value":
                if action.column and action.value:
                    if action.value == "absolute":
                        self.current_df[action.column] = pd.to_numeric(self.current_df[action.column], errors='coerce').abs()
                    else:
                        # naive generic fix value for date etc.
                        if "date" in action.column.lower():
                            self.current_df[action.column] = pd.to_datetime(self.current_df[action.column]).dt.strftime('%Y-%m-%d')
                else:
                    error = "column and value required for fix_value"
        except Exception as e:
            error = str(e)

        self.steps_taken += 1
        
        current_score = grade(self.current_df, self.expected_df)
        reward = current_score - self.last_score
        if error:
            reward -= 0.05
        
        self.last_score = current_score
        
        done = (self.steps_taken >= self.max_steps) or (current_score >= 0.99)
        
        return self.state(), reward, done, {"error": error, "score": current_score}
