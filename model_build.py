from graph import SequentialGraph
from layers import Dense, Loss


model = SequentialGraph()

model.add_layer(Dense(input_size=20, output_size=16, activation='relu'))
model.add_layer(Dense(input_size=20, output_size=8, activation='relu'))
model.add_layer(Dense(input_size=20, output_size=8, activation='softmax'))
model.add_layer(Loss(loss_function='cross_entropy'))

model.fit(X_train, y_train, lr=0.01, epochs=5)

model.predict(X_test)