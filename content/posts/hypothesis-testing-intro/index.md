+++
date = '2026-10-12T12:52:40Z'
draft = false
title = 'Introduction to Hypothesis Testing'
tags = ['Mathematics', 'Statistics']
+++
{{< katex >}}

At the start of my degree, I fell in love with mathematics because I loved
the fact that every statement can be derived and proven from axioms and definitions.
I found it deeply satisfying that everything presented by lecturers could be
proven or disproven.

This is why, for a long time, I struggled to enjoy studying statistics; since, unlike
other areas of mathematics, there is some notion of "subjectivity". One thing that
kept me coming back to it though (usually in the form of data science), is how
often it crops up amongst any STEM discipline. Statistics ultimately underpins the scientific
method, and so has natural applications from natural sciences like biology and chemistry
to social sciences like psychology and economics.

This article is going to demonstrate that one can simultaneous love formal
mathematical proofs as well as experimental study by introducing the _Hypothesis
Testing Framework_.

## Hypothesis Testing Framework
Suppose you flip a coin 100 times and observe 37 heads. What can we say about the coin?
Are these 100 flips sufficient to say that the coin is biased? Do we need to do
more testing? It is possible that either the coin is biased, or we were simply unlucky
with a fair coin. Hypothesis testing provides us with a principled way to distinguish
between these two conflicting explanations, in the same way that mathematical proof
techniques provided principled approaches to demonstrating the correctness of a statement.

In a hypothesis test, we describe the two possible outcomes formally as the
null hypothesis and the alternative hypothesis. The null hypothesis is typically
the "default" belief, e.g. in the coin scenario, our null hypothesis would be that
"the coin is fair". The alternative hypothesis is the "alternative explanation", e.g.
"the coin is biased". We typically write these in mathematical notation, denoting
the null hypothesis as \\(H_0\\) and the alternative hypothesis as \\(H_A\\). Therefore,
for the example above we would write:

\\(H_0: p = 1/2\\)

\\(H_A: p \ne 1/2\\)

where \\(p\\) denotes the probability of obtaining a head from a single coin flip.

In a general sense, the hypothesis testing framework has 3 steps:

1. Define a _null hypothesis_ (\\(H_0\\)) and an _alternative hypothesis_ (\\(H_A\\)).
2. Assuming \\(H_0\\) is true, choose a statistical model which encapsulates our observations (usually some distribution).
3. Compare our real observation with results expected from the model.
4. Using the comparison, reject \\(H\_0\\) or fail to reject \\(H_0\\).

The details of the comparison in step 3 will be the subject of the next sections.
One thing that I wish to draw attention to is the wording of "reject" and "fail to reject".
At no point do we say "accept \\(H_A\\)" or "accept \\(H\_0\\)". This is very important since
our test can only measure how compatible our observations are with \\(H_0\\), but can not
prove or disprove the null hypothesis definitively.

This is the key difference between mathematical proofs and hypothesis tests.
Mathematical proofs establish a conclusion using given premises and rules of inference;
thus if the premises are true, and the reasoning is valid, the conclusion is necessarily
true with 100% certainty. On the other hand, hypothesis tests use empirical observations
in order to assess a hypothesis under a statistical model. This means that conclusions
made by hypothesis testing are subject to some level of uncertainty.

## Confidence and Significance Levels
We define the boundary between rejection, and failing to reject \\(H_0\\) using
a _significance level_ (usually denoted by \\(\alpha\\)). This is the probability
of rejecting \\(H\_0\\) in the case that it is actually true, i.e. the probability
of an "incorrect rejection". More often than not, we use \\(\alpha = 0.05\\); values
may vary depending on application. For example, in clinical trials, we may use
a lower significance level if an incorrect rejection would be "dangerous".

We are going to statistically test whether the coin flipped in the experiment at the
start of the last section is biased with \\(\alpha = 0.05\\). Recall that we flipped a coin 100 times and
observed 37 heads. We begin by defining our hypotheses:

\\(H_0: p = \frac{1}{2}\\), i.e. the coin is fair.

\\(H_A: p \ne \frac{1}{2}\\), i.e. the coin is biased.

Assuming that the null hypothesis is true, since each coin flip is independent,
we can model the number of heads \\(X\\) achieved through the experiment using
the binomial distribution, i.e. \\(X \sim \text{Bin}\left(100, \frac{1}{2}\right)\\).

Since we are testing with \\(\alpha = 0.05\\), we want a 5% chance of "false rejection",
we can imagine this as a "2-tailed test".

**Diagram**

Thus, we can create a "rejection rule" (i.e. a rule which determines what
observations allow us to reject \\(H_0\\)) by finding the interval in which
95% of the expected outcomes lie in the distribution, this is known as the
95% _confidence interval_. 

A \\(\beta \\%\\) _confidence interval_ is a range of values which is likely to
contain the true value of a population parameter. In particular, if we repeated
the study many times, about \\(\beta \\%\\) of the intervals would contain the true value.

There are a variety of possible methods to do this: software, calculator, tables, by hand. However, after rounding
(since we can only do a integral amount of coin flips), we find that this interval
is \\(40 < x < 60\\). Then since \\(37 < 40\\), we reject \\(H_0\\) and report
that there is sufficient evidence to suggest that \\(p \ne \frac{1}{2}\\), i.e.
the coin is biased.

## Testing using \\(p\\)-values
Rather than defining a rejection rule based on confidence intervals, we can also
use \\(p\\)_-values_. **Define \\(p\\)-values**

We repeat the previous test with this technique to illustrate the difference.
Our setup is exactly the same:

\\(H_0: p = \frac{1}{2}\\)

\\(H_A: p \ne \frac{1}{2}\\)

tested with \\(\alpha = 0.05\\). Assuming the null hypothesis, we can model the number
of heads observed by experiment as \\(X \sim \text{Bin}\left(100, \frac{1}{2}\right)\\).

## Subjectivity
We end with a few remarks on the difference between a formal mathematical proof
of a fact, and these hypothesis tests.

- How does the choice of \\(\alpha\\) change things?
- What if we did proceed with \\(H_A: p < \frac{1}{2}\\)?
- Why is it impossible to have a formal mathematical proof of these sorts of things?
- Why is it good to have a standardised testing method?

## Model Choice
- Why did we use Binomial distribution?
- Can we use a different distribution?
- Does it matter?
