# Hyperbolic Bus Routes
Authors: Ruitao Yi, Abhirup Ghosh, Rik Sarkar

Get the hyperbolic embeddings for Edinburgh bus transportation network

# Requirements:
python=3.7.3 

conda install lxml numpy scipy jupyter networkx matplotlib pandas mpu joblib=0.12 gensim plotly iteration_utilities

# Steps:
1. Run Preprocessing.ipynb to process the data on the website
2. Run 4pc_distances.ipynb to get the 4pc-episilon plot
3. Run Our model.ipynb to get the hyperbolic embedding plot
