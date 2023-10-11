#!/usr/bin/env python
# coding: utf-8

# In[1]:


2+3


# In[2]:


50% 5


# In[3]:


2+10 10+3


# In[4]:


2+10 * 10+3


# In[5]:


a = 5


# In[6]:


a


# In[7]:


a=10


# In[8]:


a


# In[9]:


b=6


# In[10]:


b


# In[11]:


a + a


# In[33]:


a = a+a


# In[34]:


a


# In[35]:


a+b


# In[36]:


type(a)


# In[37]:


a=30.1


# In[38]:


type(a)


# In[39]:


my_income =100 

tax_rate =0.1

my_taxes = my_income * tax_rate


# In[40]:


my_taxes


# In[41]:


my_income= 130000


# In[42]:


my_taxes


# In[43]:


my_income =1000349 

tax_rate =0.1

my_taxes = my_income * tax_rate


# In[44]:


my_taxes


# In[45]:


import pygame
import time
import random

pygame.init()

white = (255, 255, 255)
red = (255, 0, 0)
black = (0, 0, 0)

dis_width = 800
dis_height = 600

dis = pygame.display.set_mode((dis_width, dis_height))

clock = pygame.time.Clock()

snake_block = 10
snake_speed = 15

font_style = pygame.font.SysFont(None, 50)

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width/2, dis_height/2])

def gameLoop():
    game_over = False
    game_close = False

    x1 = dis_width/2
    y1 = dis_height/2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close == True:
            dis.fill(black)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        dis.fill(black)
        pygame.draw.rect(dis, white, [foodx, foody, snake_block, snake_block])
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_List.append(snake_head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_head:
                game_close = True

        for block in snake_List:
            pygame.draw.rect(dis, white, [block[0], block[1], snake_block, snake_block])

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()


# In[46]:


import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the video
cap = cv2.VideoCapture('path_to_video_file.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for the model
    frame = cv2.resize(frame, (224, 224))
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)

    # Predict using the model
    predictions = model.predict(frame)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())

    # Display the prediction
    cv2.putText(frame, f'Prediction: {label[0][0][1]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[47]:


hello 


# In[48]:


'hello'


# In[49]:


'world'


# In[50]:


'this is also a string'


# In[51]:


"I'm going on a run"


# In[52]:


print (hello)


# In[53]:


print('hello')


# In[54]:


len('I am')


# In[55]:


mystring = 'Hello World'


# In[56]:


mystring


# In[57]:


mystring[0]


# In[58]:


mystring[8]


# In[60]:


mystring[9]


# In[61]:


mystring[-2]


# In[62]:


mystring[-3]


# In[63]:


mystring


# In[64]:


mystring= 'abcdefghijk'


# In[65]:


mystring


# In[66]:


mystring[2]


# In[67]:


mystring[2:]


# In[68]:


mystring[:3]


# In[69]:


mystring[1]


# In[71]:


mystring[3:6]


# In[72]:


mystring[1:3]


# In[74]:


mystring[::2]


# In[77]:


mystring[2:7:2]


# In[78]:


mystring[::-1]


# In[ ]:




