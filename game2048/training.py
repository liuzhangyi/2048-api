import numpy as np
import keras
import math
import random
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, concatenate, BatchNormalization, Activation
from keras.models import Model
from collections import namedtuple
from tensorboardX import SummaryWriter 
from game import Game
from expectimax._ext import board_to_move

Guide=namedtuple('Guide',('state','action'))

OUT_SHAPE=(4,4)
map_table={2**i:i for i in range(1,16)}
map_table[0]=0

def grid_ohe(arr):
	ret=np.zeros(shape=OUT_SHAPE+(16,),dtype=bool)
	for r in range(OUT_SHAPE[0]):
		for c in range(OUT_SHAPE[1]):
			ret[r,c,map_table[arr[r,c]]]=1
	return ret


class Guides:
	
	def __init__(self,capacity):
		self.capacity=capacity
		self.memory=[]
		self.position=0

	def push(self,*args):
		if len(self.memory)<self.capacity:
			self.memory.append(None)
		self.memory[self.position]=Guide(*args)
		self.position=(self.position+1)%self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def ready(self, batch_size):
		return len(self.memory)>=batch_size
	
	def __len__(self):
		return len(self.memory)



class ModelWrapper:
	
	def __init__(self,model,capacity):
		self.model=model
		self.memory=Guides(capacity)
		self.writer=SummaryWriter()
		self.training_step=0

	def predict(self,board):
		return model.predict(np.expand_dims(board,axis=0))

	def move(self, game):
		ohe_board=grid_ohe(game.board)
		'''for i in range(0,4):
			for j in range(0,4):
				if ohe_board[i][j]!=0:
					ohe_board[i][j]=math.log(ohe_board[i][j],2)'''
		suggest=board_to_move(game.board)
		direction=self.predict(ohe_board).argmax()
		game.move(direction)
		self.memory.push(ohe_board,suggest)

	def train(self,batch):
		if self.memory.ready(batch):
			guides=self.memory.sample(batch)
			X=[]
			Y=[]
			for guide in guides:
				X.append(guide.state)
				ohe_action=[0]*4
				ohe_action[guide.action]=1
				Y.append(ohe_action)
		loss,acc=self.model.train_on_batch(np.array(X),np.array(Y))
		self.writer.add_scalar('loss',float(loss),self.training_step)
		self.writer.add_scalar('acc',float(acc),self.training_step)
		self.training_step+=1



'''inputs=Input((4,4,16))
conv=inputs
FILTERS=128
conv41=Conv2D(filters=FILTERS,kernel_size=(4,1),kernel_initializer='he_uniform')(conv)
conv14=Conv2D(filters=FILTERS,kernel_size=(1,4),kernel_initializer='he_uniform')(conv)
conv22=Conv2D(filters=FILTERS,kernel_size=(2,2),kernel_initializer='he_uniform')(conv)
conv33=Conv2D(filters=FILTERS,kernel_size=(3,3),kernel_initializer='he_uniform')(conv)
conv44=Conv2D(filters=FILTERS,kernel_size=(4,4),kernel_initializer='he_uniform')(conv)

hidden=concatenate([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
x=BatchNormalization()(hidden)
x=Activation('relu')(hidden)

for width in [512,128]:
	x=Dense(width,kernel_initializer='he_uniform')(x)
	x=BatchNormalization()(x)
	x=Activation('relu')(hidden)
outputs=Dense(4,activation='softmax')(x)
model=Model(inputs,outputs)
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])'''

model=keras.models.load_model('2048.model')

modelWrapper=ModelWrapper(model,16384)



for batch_no in range(1,10000):
	game=Game()
	while game.end!=1:
		modelWrapper.move(game)
	modelWrapper.train(265)
	model.save('2048model')



class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class myAgent(Agent):

    def step(self):
	board=grid_ohe(self.game.board)
        direction = model.predict(np.expand_dims(board,axis=0)).argmax()
        return direction

















