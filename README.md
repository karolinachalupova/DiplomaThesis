# DiplomaThesis
Explaining Equity Returns with Interpretable Machine Learning. WIP.

## Way ahead 
- train networks similar to Gu et al., 2018: Empirical asset pricing via machine learning. <- I'm doing this right now, I don't think I have problems. But please see issue #4, this is crucial to decide.
- interpret them: feature importance, shapley values, other?
- see what influences interpretability to be robust vs. fragile  
  - measures of fragility: 
       - how much the interpretation differ for different random seeds? 
       - is the network suspectible to adversarial attack on interpretability as in Zou (Interpretation of Neural Networks is Fragile)? They show that a small change in input can dramatically alter interpretation.  
  - studied factors to influence interpretability: 
      - depth of neural net (0-5 hidden layers)
      - other?

