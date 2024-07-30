from pydantic import BaseModel
from typing import List
import numpy as np


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


class HashedPassword(BaseModel):
    pwd: str


class Observation(BaseModel):
    history: List[float]

class Predict(BaseModel):
    model_action_dict: dict