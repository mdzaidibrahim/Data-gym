import os
import json
import pandas as pd
from openai import OpenAI

from env.environment import DataCleaningEnv
from env.tasks import TASKS
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
    base_url=API_BASE_URL
)

def rule_based_action(obs_df):
    df = pd.DataFrame(obs_df)
    
    if df.isnull().values.any():
        return Action(action_type="drop_nulls")
    
    if df.duplicated().any():
        return Action(action_type="remove_duplicates")
    
    for col in df.columns:
        if df[col].dtype == object:
            try:
                if any(isinstance(x, str) and x.replace('.', '', 1).replace('-', '', 1).isdigit() for x in df[col]):
                    return Action(action_type="convert_type", column=col, value="float")
            except:
                pass
            
        if pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] < 0).any():
                return Action(action_type="fix_value", column=col, value="absolute")
                
    return None

def run_task(task_name):
    env = DataCleaningEnv(task_name, max_steps=8)
    obs = env.reset()
    
    print(f"[START] task={task_name} env=data_cleaning_gym model={MODEL_NAME}")
    
    done = False
    step = 0
    rewards = []
    
    last_action_str = None
    fallback_index = 0
    fallback_actions = [
        Action(action_type="drop_nulls"),
        Action(action_type="remove_duplicates"),
        Action(action_type="convert_type", column="salary", value="float"),
        Action(action_type="fix_value", column="salary", value="absolute"),
        Action(action_type="convert_type", column="date", value="str")
    ]
    
    while not done and step < 8:
        step += 1
        
        action = rule_based_action(obs.dataframe)
        
        if action is None:
            prompt = f"""
            You are an agent cleaning a dataset.
            Task Description: {obs.task_description}
            Remaining steps: {obs.remaining_steps}

            Current Dataset (JSON):
            {json.dumps(obs.dataframe, indent=2)}

            Choose one action from this JSON format:
            {{
              "action_type": "drop_nulls" | "fill_nulls" | "convert_type" | "remove_duplicates" | "fix_value",
              "column": "name of the column (optional)",
              "value": "value argument (optional, e.g. 'int', 'float', 'absolute')"
            }}

            Output ONLY valid JSON.
            """
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                raw_response = response.choices[0].message.content
                
                try:
                    action_dict = json.loads(raw_response)
                except:
                    action_dict = {"action_type": "drop_nulls"}

                action = Action(**action_dict)
            except Exception:
                action = fallback_actions[fallback_index % len(fallback_actions)]
                fallback_index += 1

        action_str = f"{action.action_type}({action.column or 'null'},{action.value or 'null'})"
        
        if action_str == last_action_str:
            action = fallback_actions[fallback_index % len(fallback_actions)]
            fallback_index += 1
            action_str = f"{action.action_type}({action.column or 'null'},{action.value or 'null'})"
            
        last_action_str = action_str
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        error = info.get("error")
        error_msg = str(error) if error else "null"
        
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
        
    score = env.last_score
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success = score >= 0.99
    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
