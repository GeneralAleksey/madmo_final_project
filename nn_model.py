from typing import Annotated
from fastapi import Depends
import numpy as np
import pandas as pd
import math

# модули библиотеки PyTorch
import torch
import torch.quantization
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn
import torch.nn.functional as F

from torch import optim
import os


import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import api_model
import api_enum



model_dict = {}

# Модель DQN (origin)
class DQN_origin(chainer.Chain):

    algorithm_name = 'DQN_origin'     # Алгоритм DQN
    epsilon = 1.0                     # Начальное значение вероятности, с которой выбирается рандомное действие
    epsilon_decrease = 1e-3           # Шаг уменьшения вероятности, с которой выбирается рандомное действие
    epsilon_min = 0.1                 # Минимальная вероятность, с которой выбирается рандомное действие
    start_reduce_epsilon = 200        # Шаг, начиная с которого начинаем уменьшать вероятность выбора рандомного действия
    gamma = 0.97                      # Коэффициент дисконтирования Q-оценки (от 0.9 до 0.99), по-умолчанию = 0.97

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN_origin, self).__init__(
            fc1 = L.Linear(input_size, hidden_size),
            fc2 = L.Linear(hidden_size, hidden_size),
            fc3 = L.Linear(hidden_size, output_size)
        )

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y

    def reset(self):
        self.zerograds()

    # Эпсилон-жадная стратегия исследования
    def get_prev_act(self, prev_obs, total_step):
        # С вероятностью эпсилон выбирается рандомное действие
        if np.random.rand() <= self.epsilon:
            prev_act = np.random.randint(3)
        else:
            # Иначе действие берется из предсказаний модели
            prev_act = self(np.array(prev_obs, dtype=np.float32).reshape(1, -1))
            # Выбирается предсказанное моделью действие с максимальным значением q-оценки
            prev_act = np.argmax(prev_act.data)

        # Уменьшаем вероятность выбора рандомного действия
        if self.epsilon > self.epsilon_min and total_step > self.start_reduce_epsilon:
            self.epsilon = self.epsilon - self.epsilon_decrease

        return prev_act

    # Расчет уравнения Беллмана
    def calc_bellman_eq(self, batch, target_dqn):
        batch_size = len(batch)
        b_prev_obs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)     # Формируем батч состояний s_t
        b_prev_act = np.array(batch[:, 1].tolist(), dtype=np.int32)                               # Формируем батч действий a_t
        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)                                 # Формируем батч вознаграждений r_t
        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)          # Формируем батч состояний s_t+1
        b_done = np.array(batch[:, 4].tolist(), dtype=np.int32)                                   # Формируем батч признаков завершения последовательности

        # Получаем Q-оценку действий в состоянии s_t от origin сетки
        # Пример Q-оценки: [5.0309963 7.9157586 3.715045 ], тут 5.0309963 - это Q-оценка действия 0 (hold) в состоянии s_t, 7.9157586 - Q-оценка действия 1 (buy), 3.715045 - Q-оценка действия 2 (sell)
        q_value_current = self(b_prev_obs)

        # Получаем Q-оценку действий в состоянии s_t+1 от target сетки
        # В каждой строке оставляем только максимальную оценку из 3-ёх предложенных оценок действий
        q_value_next = np.max(target_dqn(b_obs).data, axis=1)

        # Bellman equation
        # Q-оценка в состоянии s_t (q_value_expected) должна стремиться к дисконтированной Q-оценке состояния s_t+1 (gamma * q_value_next) + размер вознаграждения (b_reward)
        q_value_expected = copy.deepcopy(q_value_current.data)
        for j in range(batch_size):
            q_value_expected[j, b_prev_act[j]] = b_reward[j] + self.gamma * q_value_next[j] * (not b_done[j])

        return q_value_current, q_value_expected


# Модель DQN (origin + target), наследует от DQN_origin всё кроме названия)
class DQN_origin_target(DQN_origin):

    algorithm_name = 'DQN_origin_target'   # Алгоритм DQN

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN_origin_target, self).__init__(
            input_size, hidden_size, output_size
        )
        
# Модель Double_DQN (наследуется от DQN_origin)
class Double_DQN(DQN_origin):

    algorithm_name = 'Double_DQN'   # Алгоритм DQN

    def __init__(self, input_size, hidden_size, output_size):
        super(Double_DQN, self).__init__(
            input_size, hidden_size, output_size
        )

    # Расчет уравнения Беллмана
    def calc_bellman_eq(self, batch, target_dqn):
        batch_size = len(batch)
        b_prev_obs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)     # Формируем батч состояний s_t
        b_prev_act = np.array(batch[:, 1].tolist(), dtype=np.int32)                               # Формируем батч действий a_t
        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)                                 # Формируем батч вознаграждений r_t
        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)          # Формируем батч состояний s_t+1
        b_done = np.array(batch[:, 4].tolist(), dtype=np.int32)                                   # Формируем батч признаков завершения последовательности

        # Получаем Q-оценку действий в состоянии s_t от origin сетки
        # Пример Q-оценки: [5.0309963 7.9157586 3.715045 ], тут 5.0309963 - это Q-оценка действия 0 (hold) в состоянии s_t, 7.9157586 - Q-оценка действия 1 (buy), 3.715045 - Q-оценка действия 2 (sell)
        q_value_current = self(b_prev_obs)
        action_current = np.argmax(q_value_current.data, axis=1)

        # Получаем Q-оценку действий в состоянии s_t+1 от target сетки
        # В каждой строке оставляем только максимальную оценку из 3-ёх предложенных оценок действий
        q_value_next = target_dqn(b_obs).data

        # Bellman equation
        # Q-оценка в состоянии s_t (q_value_expected) должна стремиться к дисконтированной Q-оценке состояния s_t+1 (gamma * q_value_next) + размер вознаграждения (b_reward)
        # Выбор действия (action_current) выполняется origin-сеткой
        # Оценка выбранных действий выполняется taget-сеткой (q_value_next)
        q_value_expected = copy.deepcopy(q_value_current.data)
        for j in range(batch_size):
            q_value_expected[j, b_prev_act[j]] = b_reward[j] + self.gamma * q_value_next[j, action_current[j]] * (not b_done[j])

        return q_value_current, q_value_expected
    
# Модель Dueling_DQN (наследуется от Double_DQN)
class Dueling_DQN(Double_DQN):

    algorithm_name = 'Dueling_DQN'    # Алгоритм DQN

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN_origin, self).__init__(
            fc1 = L.Linear(input_size, hidden_size),
            fc2 = L.Linear(hidden_size, hidden_size),
            fc3 = L.Linear(hidden_size, hidden_size//2),
            fc4 = L.Linear(hidden_size, hidden_size//2),
            state_value = L.Linear(hidden_size//2, 1),
            advantage_value = L.Linear(hidden_size//2, output_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        hs = F.relu(self.fc3(h))
        ha = F.relu(self.fc4(h))
        state_value = self.state_value(hs)
        advantage_value = self.advantage_value(ha)
        advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
        q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))
        return q_value       

# Загрузка всех моделей с диска в память
def load_models():
    # Переводим chainer в evaluation mode для работы моделей на inference
    chainer.global_config.train = False
    chainer.config.train = False
    
    # Заполняем словарь моделей болванками
    model_dict.clear()
    
    # dqn_origin
    dqn_origin = DQN_origin(input_size=90+1, hidden_size=100, output_size=3)
    model_dict[dqn_origin.algorithm_name] = dqn_origin
    
    # dqn_origin_target
    dqn_origin_target = DQN_origin_target(input_size=90+1, hidden_size=100, output_size=3)
    model_dict[dqn_origin_target.algorithm_name] = dqn_origin_target
    
    # dqn_double
    dqn_double = Double_DQN(input_size=90+1, hidden_size=100, output_size=3)
    model_dict[dqn_double.algorithm_name] = dqn_double
    
    # dqn_dueling
    dqn_dueling = Dueling_DQN(input_size=90+1, hidden_size=100, output_size=3)
    model_dict[dqn_dueling.algorithm_name] = dqn_dueling
    
    # Десериализуем файлы моделей с диска и записываем их в память
    directoryname = 'nn_model'
    directory = os.fsencode(directoryname)   
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        f_name, f_extension = filename.split('.')
        if f_extension == 'npz' and f_name in model_dict.keys():  
            serializers.load_npz(os.path.join(directoryname, filename), model_dict[f_name])     
    

# Удаляем модели перед закрытием приложения
def clear_models():
    model_dict.clear()

# Возвращаем действия, предсказанные моделями по состоянию
def predict(history: np.array):
    predict = api_model.Predict(model_action_dict = {})
    for model_name, model in model_dict.items():
        q_values = model(history)
        predict.model_action_dict[model_name] = api_enum.ActionType(int(np.min(np.argmax(q_values.data, axis=1)))).name
    return predict
