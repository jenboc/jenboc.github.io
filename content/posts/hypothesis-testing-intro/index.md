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
- Why is hypothesis testing important?


- What does it mean for something to be "statistically significant"?

## Confidence and Significance Levels
- What is a confidence level?

- What is a significance level?

### Example
Suppose that flip a coin 100 times and observe 37 heads. We wish to statistically
test whether or not this coin is fair with \\(\alpha = 0.05\\).
We set up the following hypotheses:

\\(H_0: p = 1/2\\), i.e. the probability of the coin landing on heads is \\(1/2\\)
and the coin is thus fair.

\\(H_A: p \ne 1/2\\), i.e. the probability of the coin landing on heads is _different_
to \\(1/2\\) and the coin is thus unfair.

Since we observed less heads, one could choose that the alternative hypothesis
actually reads \\(p < 1/2\\). However, we do not wish to demonstrate that the
coin is biased in a certain direction; just that it is biased at all. Therefore,
either hypothesis is valid for what we wish to show, but they aren't equivalent
(as we shall show later).

Suppose the null hypothesis was true. Since the coin flips are independent, we
can model the experiment using a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution),
i.e. we can model the number of heads as a random variable \\(H \sim \text{Bin}(100, 1/2)\\).
We can then use pre-existing knowledge of the binomial distribution in order to calculate
our \\(1-\alpha = 0.95\\) confidence interval.

**Diagram of the distribution**

We find with 95% confidence (using a table, calculator or statisical software)
that the expected number of heads is \\(50 \pm 10\\), i.e. for a fair coin we expect
that the number of observed heads \\(h\\) satisfies \\(40 \le h \le 60\\).

In our experiment, we observed \\(37 < 40\\) heads. Therefore, we reject \\(H_0\\)
since there is sufficient evidence to suggest that \\(p \ne \frac{1}{2}\\).

## Testing using \\(p\\)-values
- What is a \\(p\\)-value?

- Why would we use a \\(p\\)-value instead of a confidence level?

### Example
- "Is this coin fair?" but with \\(p\\)-values instead

Link onto next article where we use hypothesis testing to test if board games
dice are "fair".

## Subjectivity
We end with a few remarks on the difference between a formal mathematical proof
of a fact, and these hypothesis tests.

- How does the choice of \\(\alpha\\) change things?
- What if we did proceed with \\(H_A: p < \frac{1}{2}\\)?
- Why is it impossible to have a formal mathematical proof of these sorts of things?
- Why is it good to have a standardised testing method?
