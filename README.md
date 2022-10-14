# Causing: CAUSal INterpretation using Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)

_Causing is a multivariate graphical analysis tool helping you to interpret the causal
effects of a given equation system._ 

Get a nice colored graph and immediately understand the causal effects between the variables.

**Input:** You simply have to put in a dataset and provide an equation system in form of a
python function. The endogenous variables on the left-hand side are assumed to be caused by
the variables on the right-hand side of the equation. Thus, you provide the causal structure
in form of a directed acyclic graph (DAG).

**Output:** As an output, you will get a colored graph of quantified effects acting between
the model variables. You can immediately interpret mediation chains for every
individual observation - even for highly complex nonlinear systems.

Here is a table relating Causing to other approaches:

Causing is | Causing is NOT
--- | ---
✅ causal model given | ❌ causal search
✅ DAG directed acyclic graph | ❌ cyclic, undirected, or bidirected graph
✅ latent variables | ❌ just observed / manifest variables
✅ individual effects | ❌ just average effects
✅ direct, total, and mediation effects | ❌ just total effects
✅ structural model | ❌ reduced model
✅ small and big data | ❌ big data requirement
✅ graphical results | ❌ just numerical results
✅ XAI explainable AI | ❌ black box neural network

The Causing approach is quite flexible. It can be applied to highly latent models with many of the modeled endogenous variables being unobserved. Exogenous variables are assumed to be observed and deterministic. The most severe restriction certainly is that you need to specify the causal model / causal ordering. 

## Causal Effects

Causing combines total effects and mediation effects in one single graph that is easy to explain. 

The total effects of a variable on the final variable are shown in the corresponding nodes of the graph. The total effects are split up over their outgoing edges, yielding the mediation effects shown on the edges. Just education has more than one outgoing edge to be interpreted in this way.

The effects differ from individual to individual. To emphasize  this, we talk about individual effects. And the corresponding graph, combining total and mediation effects is called the Individual  Mediation Effects (IME) graph. 

## Software

Causing is free software written in _Python 3_. Graphs are generated using _Graphviz_. See dependencies in [setup.py](setup.py). Causing is available under MIT license. See [LICENSE](LICENSE.md "LICENSE").

The software is developed by RealRate, an AI rating agency aiming to re-invent the rating market by using AI, interpretability, and avoiding any conflict of interest. See www.realrate.ai.

When starting `python -m causing.examples example` after cloning / downloading the Causing repository you will find the results  in the _output_ folder. The results are saved in SVG files. The IME files show the individual mediation effects graphs for the respective individual.

See `causing/examples` for the code generating some examples.

## Start your Model

To start your model, you have to provide the following information, as done in the example code below:

- Define all your model variables as SymPy symbols.
- Note that in Sympy some operators are special, e.g. Max() instead of max().
- Provide the model equations in topological order, that is, in order of computation.
- Then the model is specified with:
    - _xvars_: exogenous variables
    - _yvars_: endogenous variables in topological order
    - _equations_: previously defined equations
    - _final_var_: the final variable of interest used for mediation effects

## 1. A Simple Example

Assume a model defined by the equation system:

Y<sub>1</sub> = X<sub>1</sub>

Y<sub>2</sub> = X<sub>2</sub> + 2 * Y<sub>1</sub><sup>2</sup>

Y<sub>3</sub> = Y<sub>1</sub> + Y<sub>2</sub>.

This gives the following graphs. Some notes to understand them:

- The data used consists of 200 observations. They are available for the x variables X<sub>1</sub> and X<sub>2</sub> with mean(X<sub>1</sub>) = 3 and mean(X<sub>2</sub>) = 2. Variables Y<sub>1</sub> and Y<sub>2</sub> are assumed to be latent / unobserved. Y<sub>3</sub> is assumed to be manifest / observed. Therefore, 200 observations are available for Y<sub>3</sub>.

- To allow for benchmark comparisons, each individual effect is measured with respect to the mean of all observations. 

- Nodes and edges are colored, showing positive (_green_) and negative (_red_) effects they have on the final variable Y<sub>3</sub>.

- Individual effects are based on the given model. For each individual, however, its _own_ exogenous data is put into the given graph function to yield the corresponding endogenous values. The effects are computed at this individual point. Individual effects are shown below just for individual no. 1 out of the 200 observations. 

- Total effects are shown below in the nodes and they are split up over the outgoing edges yielding the Mediation effects shown on the edges. Note, however, that just outgoing edges sum up to the node value, incoming edges do not. All effects are effects just on the final variable of interest, assumed here to be Y<sub>3</sub>.

![Individual Mediation Effects (IME)](https://github.com/realrate/Causing/raw/develop/images_readme/IME_1.svg)

As you can see in the right-most graph for the individual mediation effects (IME), there is one green path starting at X<sub>1</sub> passing through Y<sub>1</sub>, Y<sub>2</sub>, and finally ending in Y<sub>3</sub>. This means that X<sub>1</sub> is the main cause for Y<sub>3</sub> taking on a value above average with its effect on Y<sub>3</sub> being +29.81. However, this positive effect is slightly reduced by X<sub>2</sub>. In total, accounting for all exogenous and endogenous effects, Y<sub>3</sub> is +27.07 above average. You can understand at one glance why Y<sub>3</sub> is above average for individual no. 1.

You can find the full source code for this example [here](https://github.com/realrate/Causing/blob/develop/causing/examples/models.py#L16-L45).

## 2. Application to Education and Wages

To dig a bit deeper, here we have a real-world example from social sciences. We analyze how the wage earned by young American workers is determined by their educational attainment, family characteristics, and test scores.

This 5-minute introductory video gives a short overview of Causing and includes this real data example: See [Causing Introduction Video](https://youtu.be/GJLsjSZOk2w "Causing_Introduction_Video").

See here for a detailed analysis of the Education and Wages example: [An Application of Causing: Education and Wages](docs/education.md).

## 3. Application to Insurance Ratings

The Causing approach and its formulas together with an application are given in:

> Bartel, Holger (2020), "Causal Analysis - With an Application to Insurance Ratings"
DOI: 10.13140/RG.2.2.31524.83848
https://www.researchgate.net/publication/339091133

Note that in this early paper the mediation effects on the final variable of interest are called final effects. Also, while the current Causing version just uses numerically computed effects, that paper uses closed formulas.

The paper proposes simple linear algebra formulas for the causal analysis of equation systems. The effect of one variable on another is the total derivative. It is extended to endogenous system variables. These total effects are identical to the effects used in graph theory and its do-calculus. Further, mediation effects are defined, decomposing the total effect of one variable on a final variable of interest over all its directly caused variables. This allows for an easy but in-depth causal and mediation analysis. 

The equation system provided by the user is represented as a structural neural network (SNN). The network's nodes are represented by the model variables and its edge weights are given by the effects. Unlike classical deep neural networks, we follow a sparse and 'small data' approach. This new methodology is applied to the financial strength ratings of insurance companies. 

> **Keywords:** total derivative, graphical effect, graph theory, do-Calculus, structural neural network, linear Simultaneous Equations Model (SEM), Structural Causal Model (SCM), insurance rating

## Award

RealRate's AI software _Causing_ is a winner of the PyTorch AI Hackathon.

<img src="https://github.com/realrate/Causing/raw/develop/images_readme/RealRate_AI_Software_Winner.png">

We are excited to be a winner of the PyTorch AI Hackathon 2020 in the Responsible AI category. This is quite an honor given that more than 2,500 teams submitted their projects.

[devpost.com/software/realrate-explainable-ai-for-company-ratings](https://devpost.com/software/realrate-explainable-ai-for-company-ratings "devpost.com/software/realrate-explainable-ai-for-company-ratings").

## Contact

Dr. Holger Bartel  
RealRate  
Cecilienstr. 14, D-12307 Berlin  
[holger.bartel@realrate.ai](mailto:holger.bartel@realrate.ai?subject=[Causing])  
Phone: +49 160 957 90 844  
[www.realrate.ai](https://www.realrate.ai "www.realrate.ai")
