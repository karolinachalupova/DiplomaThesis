# DiplomaThesis
Explaining Equity Returns with Interpretable Machine Learning. WIP.

## Motivation and Research question
There is a tradeoff between performance and interpretability. Gu et al., 2018 (Empirical asset pricing via machine learning) shows that neural networks are superior to other models in their ability to predict equity returns. However, neural networks are also the least readily interpretable models. 

Also, Zou et al. (Interpretation of neural networks is fragile) show using adversarial attack that a small change in input can dramatically alter neural network interpretation. Linear model would not be suspectible to such adversarial attack by construction, and the authors suggest that neural network's the suspectibility to attack is precisely due to their complexity. If we change input in an immaterial fashion and interpretation changes completely, the interpretability of such a model is not good.  

Also, neural networks perform well when used in ensembles. A possible way of making ensemble is to train the same model multiple times with different random seed. By interpreting same network with different seeds could be seen as another measure to asses stability of the model in terms of interpretation. Is the model's interpreation completely different just because there is a different random seed? 

Is there a sweet spot between performance and interpretability? Simpler models should be less sensitive to adversarial attacks and random differences in initialization (seed) and complex models should be more sensitive. I would like to examine this tradeoff explicitly. 


## Way ahead 
- train networks similar to Gu et al., 2018. **<- I'm doing this right now
- interpret them: feature importance, shapley values, other?
       - There is a good python library for some ML interpretation https://github.com/SeldonIO/alibi
       - I would like to use model reliance from Fisher, but I cannot find an implementation. I think the paper is super interesting if you want to have a look. https://arxiv.org/abs/1801.01489 If there is time, I can try to code it up. I understand MR but don't quite understand the MCR. But it sounds very hot: "Overall, we find that MCR provides more robust and conservative intervals for the reliance of f0 on X1 and X2, relative to standard bootstrap approaches."
       - Idea: bootstrap confidence intervals for feature importance 
- see what influences interpretability to be robust vs. fragile  
  - measures of fragility: 
       - how much the interpretation differ for different random seeds? 
       - to what degree is the network suspectible to adversarial attack on interpretability as in Zou? 
  - studied factors to influence interpretability: 
      - depth of neural net (0-5 hidden layers)
      - other?

