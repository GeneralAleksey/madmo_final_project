from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import numpy as np
import uvicorn

import api_enum
import api_model
import api_authentication
import nn_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер - ыызывается на старте и при завершении приложения"""
    # Загрузка RL-моделей с диска в память на старте приложения
    nn_model.load_models()
    yield
    # Выгрузка RL-моделей из памяти при завершении приложения
    nn_model.clear_models()

app = FastAPI(lifespan=lifespan)



@app.post("/app/v1/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> api_model.Token:
    """URL для авторизации"""
    user = api_authentication.authenticate_user(
        form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = api_authentication.create_access_token(
        data={"sub": user.username}
    )
    return api_model.Token(access_token=access_token, token_type="bearer")



@app.get("/app/v1/hash_password/{plain_password}")
def get_password_hash(plain_password: str) -> api_model.HashedPassword:
    """Возвращает хеш от заданного пароля"""
    hashed_password = api_authentication.get_password_hash(plain_password)
    return api_model.HashedPassword(pwd=hashed_password)



@app.post("/app/v1/get_next_action/", dependencies=[Depends(api_authentication.get_current_active_user)])
def get_next_action(
    previous_observation: api_model.Observation
):
    """Возвращает предсказания моделей (рекомендуемые действия)"""
    predict = nn_model.predict(np.array([previous_observation.history], dtype=np.float32))
    return predict