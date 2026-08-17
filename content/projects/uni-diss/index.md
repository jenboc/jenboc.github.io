+++
date = '2026-08-17T21:09:30+01:00'
draft = false
title = 'Using Quantum Annealing to Solve the Elliptic Subset Sum Problem'
tags = ['Quantum Computing', 'Quantum Annealing']
+++

## Summary

As part of my final year at university, I undertook a substantial research project titled
"Using Quantum Annealing to Solve the Elliptic Subset Sum Problem".

My project proposes two methods to solve certain instances of the Elliptic Subset
Sum Problem. This is a variation of the [subset sum problem](https://en.wikipedia.org/wiki/Subset_sum_problem)
which uses rational points on [elliptic curves](https://en.wikipedia.org/wiki/Elliptic_curve)
instead of integers.

The methods proposed use [quantum annealing](https://en.wikipedia.org/wiki/Quantum_annealing)
which is a form of [adiabatic quantum computation](https://en.wikipedia.org/wiki/Adiabatic_quantum_computation).
Solving a problem using quantum annealing involves reformulating the problem instance
as an instance of the [Ising Model](https://en.wikipedia.org/wiki/Ising_model) or a [Quadratic Unconstrained Binary Optimisation (QUBO) problem](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization).
This is something that can be done quite easily for the integer subset sum problem,
but is very difficult when considering rational points on elliptic curves due to
their non-linear addition rule. In other words, the project's challenge comes from
the fact that the mathematical structure of these elliptic curve points makes
it much more difficult to express the problem in a way that a quantum annealer
can work with.

Whilst the subset sum problem itself does not have many practical applications,
elliptic curves certainly do, especially in cryptography. Researchers interested
in post-quantum cryptography have recently tried to use quantum annealing in order
to solve the elliptic [discrete logarithm problem](https://en.wikipedia.org/wiki/Discrete_logarithm),
i.e. the discrete logarithm problem using rational points on elliptic curves. My project attempts to
use insights gathered by these attempts in order to solve a different computational
problem, which still involves elliptic curves.

My project discusses two potential methods of applying quantum annealing to the
elliptic subset sum problem and proves their theoretical correctness. It also
uses experimentation in order to demonstrate the practical limitations of these
methods.

The full report, including scripts used to carry out the aforementioned experiments,
are available in the git repository linked below.

## Repository

{{< github repo="jenboc/quantum-annealing-subset-sum" >}}
