from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class Observation(BaseModel):
    dataframe: List[Dict[str, Any]]
    task_description: str
    remaining_steps: int

class Action(BaseModel):
    action_type: Literal[
        "drop_nulls",
        "fill_nulls",
        "convert_type",
        "remove_duplicates",
        "fix_value"
    ]
    column: Optional[str] = None
    value: Optional[str] = None
