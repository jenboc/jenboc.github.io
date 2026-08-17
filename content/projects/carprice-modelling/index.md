+++
date = '2026-08-15T20:07:15+01:00'
draft = false
title = 'Predictive Modelling of Used Car Prices'
tags = ['Data Science', 'Python', 'Mathematics', 'Machine Learning']
+++
{{< katex >}}

## Motivation

Recently, I carried out a modelling project which used machine learning techniques
in order to predict used car prices, using dataset published on Kaggle. This
was intended as an extension of my [UK HPI analysis project](/projects/hpi-analysis)
which I started to explore the different areas of the field of data science.
This project focussed on the modelling side of data science and aimed to further
develop my exploratory analysis skills. This time, rather than seeking out and
testing hypotheses, the point of EDA was to identify features in the dataset
which could be used for prediction.

Unlike HPI analysis, used car price prediction is a fairly common data science
portfolio project. However, these projects are usually shallow and simply involve
throwing a model at the data. I wanted this project to be more profound: I wanted
to explore the dataset in greater depth, and also use the modelling section to
compare different regression approaches. Additionally, I wanted to learn how
we can use these models to further analyse the dataset.

## What I Learnt

As mentioned in my HPI project post, I had taken a few data science and applied ML
courses during my time at university. This project was more inline with my previous
learning than the data analysis project. However, due to the nature of the project,
there were still some useful lessons to be learned.

For instance, once the relevant features had been extracted from the dataset,
I found that many records appeared to represent very similar or repeated vehicles.
This made the choice of validation strategy crucial, as standard 
\\(k\\)-fold cross-validation could potentially allow closely related observations 
to appear in both the training and validation sets. Hence, I also used grouped 
\\(k\\)-fold cross-validation to assess how the models performed when these groups were 
kept separate. This gave me a better understanding of how much the apparent 
performance of the models depended on the structure of the dataset.

Moreover, one thing that I hadn't learnt in my previous studies was _model interpretation_.
This project used permutation importance in order to identify the features in the dataset
which gave the most predictive information. Much of this supported the conclusions
that I reached during exploratory analysis, particularly the importance of vehicle
age and engine power. However, there were some less intuitive results. For example,
vehicle make and model were only moderately important for price prediction.
This experience demonstrated to me that not only is model interpretation important
for explaining a model's predictions, but also for challenging initial assumptions
about the dataset.

Overall, the project improved my understanding of how data scientists develop,
evaluate and interpret machine-learning models. The full analysis and modelling
procedure, including the methodology, model comparison and results, is available
in the project repository linked below.

## Repository

{{<github repo="jenboc/used-car-price-prediction" >}}
