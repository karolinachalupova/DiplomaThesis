# DiplomaThesis
Explaining Equity Returns with Interpretable Machine Learning. WIP.

## Motivation and Research question
The question of what determines average return is widely studied. Recent years have seen an upsurge in neural networks applied to this task in academia and industry, yilding unprecedented performance. Gu et al., 2018 (Empirical asset pricing via machine learning) shows that neural networks are superior to other models in their ability to predict equity returns. However, there is a tradeoff between performance and interpretability. However, neural networks are also the least readily interpretable models. 

Model interpretability is crucial for several reasons.

The question which anomalies are important is still open. 

Neural networks perform well when used in ensembles. A possible way of making ensemble is to train the same model multiple times with different random seed. By interpreting same network with different seeds could be seen as another measure to asses stability of the model in terms of interpretation. Is the model's interpreation completely different just because there is a different random seed? ANSWER: YES, DEPENDS ON FEATURE (amount of correlation between features). This can help explain why ensembles perform so well.

Simpler models should be less sensitive to random differences in initialization (seed) and complex models should be more sensitive. I would like to examine this tradeoff explicitly. UPDATE: THE RESULT IS REALLY THERE! 

## What I have done so far
- I finished training of networks from Gu et al., 5 different depths times 9 random seeds times 5 train-valid-test splits (12-12-1, 13-12-1,14-12-1, 15-12-1, 16-12-1, numbers designate amount of years of data in given sample)
- The performance is not terrible, but R2 OOS is all over the place (see results_plots.ipynb)
- Calculated feature importance using Model Reliance and Integrated Gradients for ensembles as well as individual seeds 
- Written a little bit of thesis itself.

### Interpretation measures studied so far
- Global feature importance 
     - model reliance (Fisher) 
     - integrated gradients
- Local feature importance 
     - integrated gradients (can also show the signs)

### PRELIMINARY RESULTS: Interpretability across random seeds 
- what influences interpretability to be robust vs. fragile?  
- measure of fragility: how much the interpretation of a feature differs for different random seeds
- intuitively: if the effect of feature is robust across random seeds, we can put some confidence in the interpetation. On the other hand, if different seeds find different importnce of that feature, it is more likely that a given seed shows the feature as important simply due to chance. This points to the direction of how sure we can be about  the interpretation of the model. -> related to the statistical idea of confidence intervals.  
- Results: 
     - I identify important and unimportant features. 
     - the unimportant features are unimportant no matter what seed, time or measure of feature importance. 
     - among the set of very important features, the interpretation is not stable across random seeds. This is very interesting: 
            - it depends on feature and its degree of correlation with other important features. It is interesting to show if the stability of interpretation of two features is related to their correlation (crowding each other out depending on random seeds)
            - it depends on depth. Shallow networks have highly correlated interpretation accros random seeds, and the amount of correlation declines with depth.
     - the results hold in time and for different measures of feature importance
     - The results can help explain why ensembling works so well: different models pick up different correlated features and the truth is in between. 

## Little TODOs
IN RESULTS
- recalculate all results and replace the feature VolTrend by VolMV (will not affect the results much, but just to be 100percent correct)

IN TEXT
- go back to notation in literature review and unify it with the rest of the thesis – namely, expectations, vectors, vector elements, random variables.


## Results to calculate if there is time and interesting ideas that have been sidelined
- EASY add linear benchmark 
- DIFFICULT Dvelve deeper into Model Class Reliance (Fisher)
- DIFFICULT perform attack on interpretability from Zou et al. (Interpretation of neural networks is fragile). Zou et al. (Interpretation of neural networks is fragile) show using adversarial attack that a small change in input can dramatically alter neural network interpretation. Linear model would not be suspectible to such adversarial attack by construction, and the authors suggest that neural network's the suspectibility to attack is precisely due to their complexity. If we change input in an immaterial fashion and interpretation changes completely, the interpretability of such a model is not good.  
- More interpretability measures: 
     - DIFFICULT model fingerprint (from Beyond the Black Box paper) which shows if linear effect, nonlinear effect or interaction effects are important. 
     - EASY model linearity degree from alibi library 
     - MODERATE shapley values
     - MODERATE conditional feature importance, relative feature importance (see Relative Feature Importance paper)

