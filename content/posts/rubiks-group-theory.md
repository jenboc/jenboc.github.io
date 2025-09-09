+++
date = "2025-09-09T17:48:38+01:00"
draft = false
title = "Applying Group Theory to the Rubik's Cube"
+++
{{< katex >}}

I'm currently an undergraduate student entering their third year of study
and my favourite second year module was, without a doubt, group theory.

So in order to brush up on my group theory for the next academic year, I
decided to apply it to a problem: the Rubik's Cube. The aim of this post is
to:
1. Give some basic background on group theory for the uninitiated
2. Formalise the cube in this context
3. Calculate the number of possible cube configurations

This blog post assumes that you know what a Rubik's Cube is, and that you're
willing to power through some abstract maths. Also, if you're looking for something
absolutely mathematically rigorous... you're in the wrong place.

## What is Group Theory?
Central to a branch of mathematics known as "Abstract Algebra", group theory
studies structures known as "groups". It can be applied in many different
areas of maths such as geometry and number theory.

This group introduces some of the group theory will be using to formalise the
Rubik's Cube. We won't go into much detail here, but I do encourage those
interested to do more research.

### Definition of a Group
Let \\( G \ne \emptyset \\) be a set and \\(\star : G \times G \rightarrow G \\)
be a binary operation. We say \\( (G, \star) \\) forms a group if:
1. **Closure:** \\(G\\) is closed under \\(\star\\) (implied by the definition of \\(\star\\)), i.e.
$$\forall g,h \in G, g \star h \in G$$
2. **Associativity:** \\( \star \\) is associative, i.e. 
$$\forall g,h,k \in G, (g \star h) \star k = g \star (h \star k)$$
3. **Identity Element:** There exists \\( e \in G \\) such that \\( e \\) is an identity element for \\( \star \\), i.e.
$$\forall g \in G, e \star g = g \star e = g$$
4. **Inverse Elements:** Every element \\( g \in G \\) as an inverse \\( g^{-1} \in G \\), i.e.
$$g \star g^{-1} = g^{-1} \star g = e$$

For brevity, we will usually refer to the entire group as \\(G\\) and will omit
the symbol for the binary operation entirely, writing \\(gh\\) instead of
\\(g \star h\\).

#### Example: Integers and Addition
An example of a group we should all be familiar with is \\((\mathbb{Z}, +)\\).
We will work through each axiom informally:
1. **Closure:** the sum of any two integers gives another integer.
2. **Associativity:** if we are adding 3 integers together, the order of addition
doesn't matter.
3. **Identity Element:** for any \\(n \in \mathbb Z, n + 0 = 0 + n = n\\)
4. **Inverse Element:** for any \\(n \in \mathbb Z, n + (-n) = 0 = (-n) + n\\)

Therefore, we've shown (very briefly) that \\((\mathbb{Z}, +)\\) is a group!

#### Example: Symmetry Group
Consider a set \\(X\\). The set \\(\mathcal S_X\\) of bijective functions \\(X \rightarrow X\\)
forms a group with function composition, we call this the *symmetry group of \\(X\\)*.
This is a group since:
1. **Closure:** the composition of bijections is another bijection.
2. **Associativity:** function composition is associative.
3. **Identity Element:** the identity function is bijective.
4. **Inverse Element:** a function is bijective if and only if it is invertible.

We write \\(S_n\\) to mean the symmetry group of \\(\\{1, 2, \dots, n \\}\\).

### Subgroups
Formally, \\((H, \star)\\) is a subgroup of \\((G, \star)\\) 
(which we will write as \\(H \le G\\)) if \\(H \subseteq G\\) and \\((H, \star)\\) 
is a group in its own right.

**Important Note:** the operation in the group \\(H\\) is the same as the
operation in the group \\(G\\).

Using this definition in a proof can be cumbersome though, so instead we can
use the **subgroup criteria** which says: \\(H \le G\\) if and only if:
1. \\(H \ne \emptyset\\)
2. \\(\forall g,h \in H, gh \in H\\)
3. \\(\forall g \in H, g^{-1} \in H\\)

### Cycles in the Symmetry Group \\(S_n\\)
An \\(m\\)-cycle is an element of \\(S_n\\) which moves \\(m\\) numbers in a
structured way. For example, consider \\(a_1, \dots, a_m \in \\{1, \dots, n\\}\\).

{{< mermaid >}}
flowchart LR
    a1["a₁"] --> a2["a₂"]
    a2 --> a3["a₃"]
    a3 --> a4["a₄"]
    a4 --> a5["..."]
    a5 --> am["aₘ"]
    am --> a1
{{< /mermaid >}}

Mathematically, we would write this as 
\\(
\begin{pmatrix}
    a_1 & a_2 & \cdots & a_m
\end{pmatrix}
\\).

One property of \\(m\\)-cycles which we will discuss is _parity_. All cycles can
be written as a product of transpositions (2-cycles). So we say that an
\\(m\\)-cycle is **odd** if it can be written as a product of an **odd** number
of transpositions; and **even** if it can be written as a product of an **even**
number of transpositions.

## Formalising the Rubik's Cube
Now, let's apply some of the group theory discussed above in the context of a
Rubik's Cube puzzle.

Intuitively, we know that any physically possible move on a Rubik's cube can be
done by doing some sequence of quarter turns on any of the faces. Notice that:
1. If we flip one of the edge cubies (i.e. a white edge is showing the wrong
sticker on the white face), then another of the edge cubies must be flipped too.
In other words, an even number of edges must be flipped.
2. We can say a similar thing about the corners, but instead the number of
flipped corners should be a multiple of 3.

These facts aren't particularly important right now, but will be important
when we come to calculate the number of cube configurations.

So let's start describing the group, let's call it \\(G\\), by considering moves on the cube as a
permutation of the _stickers_. Then we must have that \\(G \le S_{48}\\) 
(since there are 48 moveable stickers on the cube).

As we said before, any possible move is some combination of the quarter turns. 
Formally, we would say that G is generated by f, b, u, d, l, r (corresponding
to each of the basic moves). This means for any move \\(g \in G\\) we have that
$$
    g = f^n \cdot b^m \cdot u^x \cdot d^y \cdot l^z \cdot r^w
$$
for some \\(n, m, x, y, z, w \in \mathbb Z\\).

Any one of these quarter moves would permute the 20 stickers which make up the
face in 5 4-cycles (2 for the edge stickers, 3 for the corner stickers). So the
total permutation for just the edge stickers is odd, and so is the total
permutation for just the corner stickers.

It can be then shown that, for any \\(g \in G\\), the total permutation of edges
has the same parity (odd or even) as the total permutation for corner stickers.

## How many possible cube configurations are there?
Let's proceed with calculating the number of possible cube configurations.
For this we must consider our restrictions on edge flips and corner twists.

Recall that an even number of the 12 edge cubies must be flipped in any possible
cube configuration. This means if we were going to make up a cube state, we would be 
able to decide freely whether 11 of the 12 edges are flipped; the state of the
12th edge would be determined from this.

Similarly for the corner cubies, we could decide how 7 of the 8 corners are
twisted and the state of the 8th corner would be determined from this.

Also, recall that the parity of the edge and corner permutations must match,
i.e. both permutations must be odd, or both must be even.

This brings our final configuration count to
$$
    |G| = \frac{3^7 \cdot 8! \cdot 2^{11} \cdot 12!}{2} 
    = 43,252,003,274,489,856,000
$$
