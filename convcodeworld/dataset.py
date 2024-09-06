from pydantic import BaseModel, Field
from dspy.datasets.dataset import Dataset
from datasets import load_dataset

class BigCodeBench(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        dataset = load_dataset("bigcode/bigcodebench")
        self._test = dataset['v0.1.0_hf']

class InitInput(BaseModel):
    context: str = Field(description="Related code")
    query: str = Field(description="The initial problem description from user")

class Output(BaseModel):
    code: str = Field(description="The code prediction")

