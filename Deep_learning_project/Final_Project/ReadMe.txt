
The three machine learning models that I have used are based on Neural network and genetic algorithm. These are as follows:

1. L-system based NE 
2. heuristics based NE 

Both these models are initiated with a simple neural network architecture and becomes more complex organism/architecture from 1 generation to another where the fittest organism's genomes/weights are passed to next generation. Each generation has their own unique topology size with a population (tournament selection) who are made to complete within their own class/niche. Not allowing species to compete against other topologies is called protecting innovation through speciation. Both the techniques have connection constraints in hidden and input layers.

For comparison, I have used a model which is as follows:
3. traditionally used Neuroevolution technique that has fixed topology without connection penalty constraints, ie. It can use as many hidden neurons or layers as possible for convergence. 

I have used all three techniques in construction of an intrusion detection system by using AWID dataset and compare the best performing neural network architectures. Results show, NEs with penalty constraints show faster and stable convergence in comparison with the NE without connection constraints.
