# DiplomaThesis
Explaining Equity Returns with Interpretable Machine Learning. WIP.

## Motivation and Research question
There is a tradeoff between performance and interpretability. Gu et al., 2018 (Empirical asset pricing via machine learning) shows that neural networks are superior to other models in their ability to predict equity returns. However, neural networks are also the least readily interpretable models. 

Also, Zou et al. (Interpretation of neural networks is fragile) show using adversarial attack that a small change in input can dramatically alter neural network interpretation. Linear model would not be suspectible to such adversarial attack by construction, and the authors suggest that neural network's the suspectibility to attack is precisely due to their complexity. If we change input in an immaterial fashion and interpretation changes completely, the interpretability of such a model is not good.  

Also, neural networks perform well when used in ensembles. A possible way of making ensemble is to train the same model multiple times with different random seed. By interpreting same network with different seeds could be seen as another measure to asses stability of the model in terms of interpretation. Is the model's interpreation completely different just because there is a different random seed? 

Is there a sweet spot between performance and interpretability? Simpler models should be less sensitive to adversarial attacks and random differences in initialization (seed) and complex models should be more sensitive. I would like to examine this tradeoff explicitly. 


## Way ahead 
### Train networks similar to Gu et al., 2018. 
- I finished training 5 different depths times 9 random seeds times 5 train-valid-test splits (12-12-1, 13-12-1,14-12-1, 15-12-1, 16-12-1, numbers designate amount of years of data in given sample)
- The performance is not terrible, but R2 OOS is all over the place (see results_plots.ipynb)

### Interpretation <- I'm doing this right now
- Global feature importance 
     - model reliance (Fisher) - see issues, SOMEWHAT DONE
     - integrated gradients SOMEWHAT DONE
     - other?
- Local feature importance 
     - shapley values?
     - integrated gradients?
     - other?
- Other? 
- (for SOMEWHAT DONE, see results_plots.ipynb)

### Interpretability determinants
- see what influences interpretability to be robust vs. fragile  
  - measures of fragility: 
       - how much the interpretation differ for different random seeds? 
       - to what degree is the network suspectible to adversarial attack on interpretability as in Zou? 
  - studied factors to influence interpretability: 
      - depth of neural net (0-5 hidden layers)
      - other?

