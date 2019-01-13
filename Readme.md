
# Objective Optimization
## A scipy.optimize.minimize wrapper


```python
# Loading module and starting notebook printing
import Objective_Optimization as OO
OO.init_notebook()
```


```python
OO_obj = OO.Objective_Optimization() # Crateing OO instance
```


```python
# Adding variables and parameters
# Optimization variables
OO_obj.add_opt_var("x") 
OO_obj.add_opt_var("y")
# Parameters (fixed variables for the optimizer)
OO_obj.add_opt_var("a",value=1)
OO_obj.add_opt_var("b",value=2)
```


```python
# Adding objective function (In this case the Rosenbrook function)
OO_obj.add_expression_objective_function((OO_obj.a-OO_obj.x)**2+OO_obj.b*(OO_obj.y-OO_obj.x**2)**2)
```


```python
# Printing out the OO's repesentation of the problem 
OO_obj
```




$$\underset{ x, y, a, b }{ \text { minimize } }\quad b \left(- x^{2} + y\right)^{2} + \left(a - x\right)^{2}\quad\quad\quad \text{with}\quad \begin{gather} x \in \left [ -\infty, \quad \infty\right ]\\ y \in \left [ -\infty, \quad \infty\right ]\\ a \in \left [ -\infty, \quad \infty\right ]\\ b \in \left [ -\infty, \quad \infty\right ] \end{gather}$$




```python
# Changing "a" and "b" to parameters with fixed value
OO_obj.a.set_passive()
OO_obj.b.set_passive()
OO_obj
```




$$\underset{ x, y }{ \text { minimize } }\quad \left(- x + 1\right)^{2} + 2 \left(- x^{2} + y\right)^{2}\quad\quad\quad \text{with}\quad \begin{gather} x \in \left [ -\infty, \quad \infty\right ]\\ y \in \left [ -\infty, \quad \infty\right ] \end{gather}$$




```python
# Running the optimization
OO_obj.optimize()
```
