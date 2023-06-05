import uvicorn
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Path, Query
import tensorflow as tf
import numpy as np
import pandas as pd
import os

app = FastAPI()  # create a new FastAPI app instance

# Define a Pydantic model for an item
class Item(BaseModel):
    userID: int


model = tf.keras.models.load_model('./pleasework')
df_enrollments = pd.read_csv('Dataset_Recommendation_System_-_Enrollment_Dataset_enroll_only.csv',  dtype={'user_id': 'str', 'task_id':'str'})

def delete_enrolled_tasks(userid, titles):
  titles = titles.tolist()
  recommended_titles = []
  for i in range(len(titles[0])):
    has_been_enrolled = False
    cleaned_title = str(titles[0][i]).replace('b\'', '')
    cleaned_title = cleaned_title.replace('\'', '')
    for task_enrolled in df_enrollments[df_enrollments['user_id']==str(userid)].task_title:
      if cleaned_title == task_enrolled:
        has_been_enrolled = True
        break
    if not has_been_enrolled: 
      recommended_titles.append(cleaned_title)
      print(cleaned_title)
      #print('a')
  return recommended_titles


@app.get("/")
def hello_world():
    return ("hello world")


@app.post("/")
def add_item(item: Item):
    _, title = model.predict([str(item.userID)])
    result = delete_enrolled_tasks(item.userID, title)
    return result