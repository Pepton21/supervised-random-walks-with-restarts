# Supervised Random Walks with Restarts
## Introduction
This repository holds an implementation of the **Supervised Random Walks with Restarts** algorithm proposed by **Jure Leskovec** and **Lars Backstorm** [1]. This algorithm was used in my Bachelor's thesis on the topic of *"Function Recommendation with Supervised Random Walks with Restarts on Function Call Graphs"*. The goal of the thesis is to find functions that are in some way **connected** in a given **software package** (in my thesis the **Apache APR submodule**), so that a user can be **recommended functions** that are akin to a given **query function**. This can be useful as a workaround when software packages have a **lack of documentation**.

## Algorithm description

To start with, since the algorithm is supervised, a training set is needed with examples that are labeled as positive or negative. The set of positive examples for a node ***s*** is denoted by ***D = { d1, d2, ... , dk }*** and is consisted of functions that are similar or related to *s*. The set ***L = { l1, l2, ... , ln }*** is the set of negative test examples (functions that are not similar and not related to *s*). The set ***C=D ∪ L*** is the set of recommendation candidates for node *s*. The set ***N*** is the set of all nodes.

Next, an edge strength function ***f<sub>w</sub>(Ψ<sub>ij</sub>)*** needs to be defined which calculates the strength of the edge between nodes *i* and *j*, denoted by **a<sub>ij</sub>**, with respect to their attribute/feature vector **Ψ<sub>ij</sub>**, which is a vector formed from the union of the node and edge attributes concerning edge **(i,j)**. This function is parametrized by the vector ***w*** of weights for each attribute vector element. Since the vector ***w*** may contain negative and zero values, the edge strength function should always return positive values, to avoid zero divisions and the occurrence of negative probabilities in the algorithm in the future. A good choice would be to use **exponential** edge strength:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$a_{ij}&space;=&space;f_w(\psi_{ij})&space;=&space;\exp\{&space;\psi_{ij}\cdot&space;w&space;\}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$a_{ij}&space;=&space;f_w(\psi_{ij})&space;=&space;\exp\{&space;\psi_{ij}\cdot&space;w&space;\}$$" title="$$a_{ij} = f_w(\psi_{ij}) = \exp\{ \psi_{ij}\cdot w \}$$" /></a>
  </p>

However, during experimentation it was established that using this function proved to prolong execution time by a significant margin, which is why a more simple edge strength function was used in the form of a simple dot product:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$a_{ij}&space;=&space;f_w(\psi_{ij})&space;=&space;\psi_{ij}\cdot&space;w&space;&plus;&space;1,\mbox{&space;}&space;w\geq&space;0$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$a_{ij}&space;=&space;f_w(\psi_{ij})&space;=&space;\psi_{ij}\cdot&space;w&space;&plus;&space;1,\mbox{&space;}&space;w\geq&space;0$$" title="$$a_{ij} = f_w(\psi_{ij}) = \psi_{ij}\cdot w + 1,\mbox{ } w\geq 0$$" /></a>
 </p>

Using this edge strength function adds an additional constraint for *w* stating that the parameter vector must be nonnegative. This is needed to avoid getting negative edge strengths, which will lead to negative transition probabilities in the future. Note that other edge strength functions can be used as well; however, the edge strength function must be differentiable.

The supervised step is to learn a minimal function ***F(w)*** that assigns weights *w* such that the probability of the random walk to end up in the set of positive nodes ***D*** is **maximized**. To formulate the problem, several elements are defined. The matrix **A={a<sub>ij</sub>}<sub>NxN</sub>**, obtained by applying the edge strength function to the feature/attribute vectors of all edges, is called the edge strength matrix. The matrix ***Q'={q'<sub>ij</sub> }<sub>NxN</sub>*** is the transition probability matrix, calculated from the edge strength matrix in the following way:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$q'_{ij}&space;=&space;\begin{cases}&space;\frac{a_{ij}}{\sum_k&space;a_{ik}},&space;&&space;\mbox{if&space;}(i,j)\in&space;E&space;\\0,&space;&&space;\mbox{otherwise}&space;\end{cases}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$q'_{ij}&space;=&space;\begin{cases}&space;\frac{a_{ij}}{\sum_k&space;a_{ik}},&space;&&space;\mbox{if&space;}(i,j)\in&space;E&space;\\0,&space;&&space;\mbox{otherwise}&space;\end{cases}$$" title="$$q'_{ij} = \begin{cases} \frac{a_{ij}}{\sum_k a_{ik}}, & \mbox{if }(i,j)\in E \\0, & \mbox{otherwise} \end{cases}$$" /></a>
 </p>

The random walks are with restarts, which means that in each step there is a probability ***α*** for the walker to return to the start node of the graph. This probability modifies the transition probability matrix in the following way:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$q_{ij}&space;=&space;(1-\alpha)q'_{ij}&space;&plus;&space;\alpha&space;1&space;(j&space;=&space;s)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$q_{ij}&space;=&space;(1-\alpha)q'_{ij}&space;&plus;&space;\alpha&space;1&space;(j&space;=&space;s)$$" title="$$q_{ij} = (1-\alpha)q'_{ij} + \alpha 1 (j = s)$$" /></a>
</p>

To find the probability that the random walker will end up in one of the nodes, we need to compute the stationary distribution of the stochastic process described by the matrix ***Q***. The stationary distribution of the transition probability matrix is defined by the following system of equations in matrix form:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$p^T&space;=&space;p^TQ$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p^T&space;=&space;p^TQ$$" title="$$p^T = p^TQ$$" /></a>
</p>

Having described all of these measures, the goal function ***F(w)*** that needs to be minimized is defined as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$F(w)&space;=&space;||w||^2&space;&plus;&space;\lambda&space;\sum_{d\in&space;D,&space;l&space;\in&space;L}&space;h(p_l&space;-&space;p_d)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$F(w)&space;=&space;||w||^2&space;&plus;&space;\lambda&space;\sum_{d\in&space;D,&space;l&space;\in&space;L}&space;h(p_l&space;-&space;p_d)$$" title="$$F(w) = ||w||^2 + \lambda \sum_{d\in D, l \in L} h(p_l - p_d)$$" /></a>
</p>

Note that the function ***h(x)*** is a loss function that penalizes mispredictions and ***λ*** is a regularization parameter. The loss function must be differentiable for the minimizer to work, so in this case we will use the Wilcoxon-Mann-Whitney loss function defined as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$h(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x/b}}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$h(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x/b}}$$" title="$$h(x) = \frac{1}{1 + e^{-x/b}}$$" /></a>
</p>

This function is very popular in **gradient descent approaches** due to its good mathematical properties. For ***b=1*** we obtain the widely used sigmoid function, which will be used in the experiments. The work done in [1] shows that the regularization parameter ***λ*** does not play a significant role in the algorithm and that ***λ = 1*** yields the best performance, which is why this parameter will be left out of the equation.
The vector ***p<sup>T</sup>*** is the connection between the matrix ***Q*** and the parameters ***w*** of the function that is being minimized. To find the optimal vector *w*, the **derivative** of the **goal function** needs to be provided to the **minimizer**. For a more clear representation, the variable ***δ<sub>ld</sub> = pl-pd*** is introduced. The derivative of the goal function is calculated as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$\frac{\partial&space;F(w)}{\partial&space;w}&space;=&space;2w&space;&plus;&space;\sum_{l,d}&space;\frac{\partial&space;h(p_l&space;-&space;p_d)}{\partial&space;w}&space;=&space;2w&space;&plus;&space;\sum_{l,d}&space;\frac{\partial&space;h(\delta_{ld})}{\partial&space;\delta_{ld}}(\frac{\partial&space;p_l}{\partial&space;w}-\frac{\partial&space;p_d}{\partial&space;w})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\frac{\partial&space;F(w)}{\partial&space;w}&space;=&space;2w&space;&plus;&space;\sum_{l,d}&space;\frac{\partial&space;h(p_l&space;-&space;p_d)}{\partial&space;w}&space;=&space;2w&space;&plus;&space;\sum_{l,d}&space;\frac{\partial&space;h(\delta_{ld})}{\partial&space;\delta_{ld}}(\frac{\partial&space;p_l}{\partial&space;w}-\frac{\partial&space;p_d}{\partial&space;w})$$" title="$$\frac{\partial F(w)}{\partial w} = 2w + \sum_{l,d} \frac{\partial h(p_l - p_d)}{\partial w} = 2w + \sum_{l,d} \frac{\partial h(\delta_{ld})}{\partial \delta_{ld}}(\frac{\partial p_l}{\partial w}-\frac{\partial p_d}{\partial w})$$" /></a>
</p>

The calculation of this derivative is **not trivial** due to recursive dependencies of the ***partial derivatives of p*** with respect to *w*; however, the work presented in [1] provides an efficient iterative algorithm to calculate the partial derivatives:

<p align="center">
<img src="https://github.com/Pepton21/Supervised-random-walks-with-restarts/blob/master/images/PageRank.PNG" width="400" alt="PageRank">
</p>

The only thing left that is needed to calculate the **partial derivatives** of the **stationary distribution** is the derivative of the transition probability matrix *Q*, which is easily obtained by simple derivation of the previous definition:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$\frac{\partial&space;Q_{ju}}{\partial&space;w}&space;=&space;(1-\alpha)\frac{&space;\frac{\partial&space;f_w(\psi_{ju})}{\partial&space;w}(\sum_k&space;f_w(\psi_{jk}))&space;-&space;f_w(\psi_{ju})(\sum_k&space;\frac{\partial&space;f_w(\psi_{jk})}{\partial&space;w})&space;}{&space;(\sum_k&space;f_w(\psi_{jk})&space;)^2&space;}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\frac{\partial&space;Q_{ju}}{\partial&space;w}&space;=&space;(1-\alpha)\frac{&space;\frac{\partial&space;f_w(\psi_{ju})}{\partial&space;w}(\sum_k&space;f_w(\psi_{jk}))&space;-&space;f_w(\psi_{ju})(\sum_k&space;\frac{\partial&space;f_w(\psi_{jk})}{\partial&space;w})&space;}{&space;(\sum_k&space;f_w(\psi_{jk})&space;)^2&space;}$$" title="$$\frac{\partial Q_{ju}}{\partial w} = (1-\alpha)\frac{ \frac{\partial f_w(\psi_{ju})}{\partial w}(\sum_k f_w(\psi_{jk})) - f_w(\psi_{ju})(\sum_k \frac{\partial f_w(\psi_{jk})}{\partial w}) }{ (\sum_k f_w(\psi_{jk}) )^2 }$$" /></a>
</p>

With the **goal function** and its **gradient** defined, everything is ready for a gradient descent approach to be used to minimize the goal function **F(w)**. Using the parameter vector ***w*** that minimizes the goal function and which tells us which edge attributes are important, we can calculate the PageRanks ***p*** and recommend the functions with the **highest scores**.

## Running the algorithm on Apache (Methodology)

To run the algorithm on a given software package, we must first **extract** the ***caller/callee interactions*** between functions. In this example, the free tool **CFlow** [2] was used, which uses the **Abstract Syntax Tree** to extract this information. The output of Cflow is then parsed to form a **directed graph** data structure where the nodes are functions and the directed edges are *caller/callee* interactions. The package contains **test functions** that are used as the testing dataset. We run the Supervised Random Walks with Restarts many times and calculate the **mean** of all obtained weight vectors as the final result. The **architecture** is given in the following figure:

<p align="center">
<img src="https://github.com/Pepton21/Supervised-random-walks-with-restarts/blob/master/images/architecture.PNG" width="650" alt="architecture">
</p>

## Comparison with an existing function recommendation algorithm (FRAN)

The performances of **FRAN** [3] and **SRW with restarts** are compared using the ***F1 measures***. In order to have a reliable comparison, the two algorithms need to be evaluated on the **same data**. The work presented in [3] is performed on an older version on **Apache**. This is not a problem, since the FRAN source code is publicly available and can be run on any version of Apache. In this case, both algorithms are ran on the function call graph of the **Apache httpd 2.0.65**.

Since the entire function **call graph** is too large to draw, the edge strengths obtained this way can be used to represent the call graph through a **density matrix**:

<p align="center">
<img src="https://github.com/Pepton21/Supervised-random-walks-with-restarts/blob/master/images/density_map.PNG" width="600" alt="density">
</p>

The most **dense pixels** are concentrated around the **diagonal** of the matrix, which means that nodes that are close to each other in the function call graph (have a similar node id) interact the **strongest** (have the largest edge strengths). It is interesting to observe that there are also dense pixels concerning the function with id *700*. This is actually due to the fact that function 700 is an *assert* function used in a lot of test functions.
 
The top *10* functions are considered to calculate the *F1* measures of both FRAN and the SWR with Restarts algorithm. The results are shown below:

<p align="center">
<img src="https://github.com/Pepton21/Supervised-random-walks-with-restarts/blob/master/images/F1_comparison.PNG" width="600" alt="F1">
</p>

The plots are divided into **three segments**. The leftmost segment is represented by the functions for which the SRW with Restarts algorithm gives a **better** *F1* measure. The middle segment is consisted of the functions for which the two algorithms produce the **same** *F1* measures. The rightmost segment is represented by the functions for which FRAN performs better. It can be seen that the SRW algorithm **outperforms** FRAN in most cases. SRW is **better in *209* cases**, FRAN performs better in *47* cases and they are tied in *48* cases. The tied score between the two algorithm, in most cases, is due to them both having a **zero** *F1* measure for some functions, which means that both algorithms **fail** to give **relevant recommendations**. The maximal value of the *F1* measure is also in the favor of the SRW algorithm (*0.5333* compared with the maximal *0.45* achieved by FRAN). On top of that, the average value of the *F1* measure of SRW is also higher (*0.128673* compared to FRAN's value of *0.051625*).

## References

1. Backstrom L. and Leskovec J. Supervised random walks: predicting and recommending links in social networks. Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011. Web page retrieved 23/06/2017 at http://dl.acm.org/citation.cfm?id=1935914.

2. Poznyakoff, Sergey. GNU cflow. (2005). Web page retrieved 27/06/2017 at https://www.gnu.org/software/cflow/manual/cflow.pdf.

3. Saul, Zachary M., et al. Recommending random walks. Proceedings of the the 6th joint meeting of the European software engineering conference and the ACM SIGSOFT symposium on The foundations of software engineering. ACM, 2007. Web page retrieved 23/06/2017 at http://dl.acm.org/citation.cfm?id=1287629.
