# Training Linear Recurrent Units

Implementation of the training process of Linear Recurrent Units defined from Orvieto et al., 2023, (https://arxiv.org/abs/2303.06349)

The version of the library used for the experiments:<br>
- numpy: 2.0.2<br>
- jax: 0.4.35<br>
- flax: 0.10.2<br>
- tensorflow: 2.18.0<br>
- optax: 0.2.3<br>

### Contents of the repository :
- lru.png: Scheme of the model<br>
#### 1. Training the model<br>
- training.ipynb: Jupyter notebook training the model <br>
- data_transformation.py: Python code reshaping the training input data, and creating dictionaries of parameters for the optimizer <br>
- LRU.py: Implementation of the Linear Recurrent Units from Orvieto et al., 2023, (https://arxiv.org/abs/2303.06349)<br>
- final_model.py: Python code defining the final nnx module, combining Linear Recurrent Units and Multilayer Perceptrons, and building blocks<br>
#### 2. Layer-wise parameterization:<br>
- training_layer.ipynb:Jupyter notebook training the model with layer-wise parameterization<br>
- layer_parameterization.py: Python codes containing new definition of linear layers for layer-wise parameterization<br>
- final_model_layer.py: Python code defining the final nnx module, adding layer-wise parameterization