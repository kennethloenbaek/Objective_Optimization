# Loading modules
import sympy as sym
import scipy.optimize as opt
import numpy as np
from collections import OrderedDict 

def init_notebook():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
    sym.init_printing()


class Objective_Optimization:
    def __init__(self, opt_type="min"):
        self.opt_type = opt_type
        self.opt_var = np.array([])
        self.obj_fun = None
        self.con_fun = np.array([])        
        
    # Variable methods ------------------------------------------------------------------------------------- #
    def add_optimization_variable(self, name, value=0, bound = [-np.inf,np.inf], sym_name=None, active=True):
        '''Adding variable to Optimization Objective
        
        Args:
            name: str (Name for the variable, to set display name use sym_name)
            value: int/float (Default value and initial value for optimization)
            bound: [lower,upper] (Bounds for the varible)
            sym_name: str (name to be displayed - using latex, without $$)
            active: bool (If False it is used as a parameter - this is the ONLY difference between parameter and variable)
        '''
        self.opt_var = np.append(self.opt_var,optimization_variable(name, value, bound, sym_name, active))
        setattr(self,name,self.opt_var[-1])    
        
    add_opt_var = add_optimization_variable

    def add_optimization_parameter(self, name, value=0, bound = [-np.inf,np.inf], sym_name=None, active=False):
        self.add_optimization_variable(name = name, value = value, bound=bound, sym_name = sym_name, active = active)
        
    def get_active_var(self):
        return self.opt_var[[var.active for var in self.opt_var]]
    
    def get_passive_var(self):
        logic_array = [not(var.active) for var in self.opt_var]
        return self.opt_var[logic_array]
    
    def get_active_var_names(self):
        return ", ".join([sym.latex(var) for var in self.get_active_var()])
    
    def get_active_var_bound_latex(self):
        str_out = ""
        act_var = self.get_active_var()
        for var in act_var:
            str_out += r"{} \in {}\\ ".format(sym.latex(var.name), sym.latex(var.bound))
        return str_out[:-3].replace("inf",r"\infty")
    
    # Objective Function methods --------------------------------------------------------------------------- #
    def add_expression_objective_function(self, expr, com_jac=True):
        self.obj_fun = expression(expr, com_jac)
        
    def add_objective_function(self, fun, jac=None, sym_name=None):
        self.obj_fun = function(fun, self.opt_var, jac, sym_name)  
        
    # Constraints Function methods ------------------------------------------------------------------------- #
    def add_expression_constraint_function(self, expr, con_type=">", com_jac=True, active=True):
        self.con_fun = np.append(self.con_fun,expression_constraint(expr, con_type, com_jac, active)) 
        
    def add_constraint_function(self, fun, jac=None, sym_name=None):
        self.con_fun = np.append(self.con_fun,function_constraint(fun, self.opt_var, jac=None, sym_name=None)) 
        
    def get_active_con(self):
        return self.con_fun[[con.active for con in self.con_fun]]
    
    def get_passive_con(self):
        return self.con_fun[[not(con.active) for con in self.con_fun]]
    
    # Convert between Obj. Opt. and scipy ------------------------------------------------------------------ #
    def _set_index2var(self):
        '''Array index to dict name'''
        act_var = self.get_active_var()
        self._index2var = [vari.name_in for vari in act_var]
    
    def obj_fun_eval(self,x):
        '''Objective function to be used by scipy'''
        # array2dict values
        self._set_curr_val(x)
        # Evaluate user obj_function
        obj_val = self.obj_fun(**self.curr_var)
        return self.obj_sign*obj_val
    
    def _set_curr_val(self,x):
        for vari_name, val in zip(self._index2var,x) :
            self.curr_var[vari_name] = val 
    
    def _obj_jac_fun(self,x):
        self._set_curr_val(x)
        return self.jac(**self.curr_var)
    
    def _get_bounds(self):
        '''Bounds to list (accouding to array index)'''
        return [getattr(self,name).bound for name in self._index2var]
    # Constraint to dict
    
    # Optimize methods ------------------------------------------------------------------------------------- #
    def optimize(self, verbose=True):
        '''Start optimization'''
        
        self._set_index2var()
        self.curr_var = {opt_var.name_in: opt_var.value for opt_var in self.opt_var}
        x0 = [self.curr_var[name] for name in self._index2var]
        self.jac = self.obj_fun.get_jac(self)
        jac = self._obj_jac_fun if self.jac else None
        bound = self._get_bounds()
        cons = None # get constraints dict
        self.obj_sign = -1 if self.opt_type is "max" else 1                   
        self.scipy_res = opt.minimize(fun = self.obj_fun_eval, x0=x0, jac = jac, bounds=bound)
        
    # Mic. methods ----------------------------------------------------------------------------------------- #
    def _repr_html_(self):
        opt_type = "minimize" if self.opt_type == "min" else "maximize" # Name for optimization
        opt_var = self.get_active_var_names() # Active variables
        opt_prob = sym.latex(self.obj_fun.get_opt_prob(self.get_passive_var())) # Optimization problem 
        
        # Opt problem 
        opt_prob = r"\underset{{ {} }}{{ \text {{ {} }} }}\quad {}".format(opt_var,opt_type,opt_prob)
        
        # Bounds
        opt_bounds = r"\quad\quad\quad \text{with}\quad \begin{gather} %s \end{gather}"%(self.get_active_var_bound_latex())
        
        # Constraints
        opt_con = ""
        if self.get_active_con().any():
            opt_con += r"\\ \text{Subject to}\quad\quad"
            opt_con += r" \begin{array}{c}"
            for con in self.get_active_con():
                opt_con += r"%s %s 0 \\"%(sym.latex(con.get_opt_prob(self.get_passive_var())),con.con_type)
            opt_con = opt_con[:-2]
            opt_con += r"\end{array}"
        
        return r"$$"+opt_prob+opt_bounds+opt_con+"$$"

class optimization_variable(sym.Symbol):

    def __new__(cls, name, value=0, bound = [-np.inf,np.inf], sym_name=None, active=True,**kwargs):
        if sym_name is None:
            sym_name = name            
        return super().__new__(cls,sym_name,**kwargs)
    
    def __init__(self, name, value=0, bound= [-np.inf,np.inf], sym_name=None, active=True):
        self.name_in = name
        self.value = value
        self.default_value = value
        self.bound = bound
        self.active = active
    
    def __str__(self):
        return r"({}, val:{}, def_val:{}, bond:{}, active:{})".format(self.name_in, self.value, self.default_value, self.bound, self.active)
        
    def set_value(self,value):
        self.value = value
    
    def set_default_value(self,default_value):
        self.default_value = default_value
        
    def set_bound(self, bound):
        self.bound = bound
        
    def set_passive(self):
        self.active = False
        
    def set_active(self):
        self.active = True
        
class function:    
    def __init__(self, fun, var, jac=None, sym_name=None): 
        if not(sym_name):
            sym_name = fun.__name__
        self.fun = fun
        self.sym = sym.Function(sym_name)(*var)
        self.jac = jac
    
    def get_opt_prob(self, passive_var):
        opt_prob = self.sym.copy()
        for var in passive_var:
            opt_prob = opt_prob.subs(var, "%s=%s"%(var.name,var.value))
        return opt_prob
    
    def get_jac(self,*args):
        return self.jac
    
    def __call__(self,**kwargs):
        return self.fun(**kwargs)

class function_constraint(function):
    def __init__(self, fun, var, con_type=">", jac=None, sym_name=None, active=True):
        self.active = active
        self.con_type = con_type
        super().__init__(fun, var, jac, sym_name)
        
    def set_passive(self):
        self.active = False
        
    def set_active(self):
        self.active = True
        
class expression:
    def __init__(self, expr, var, com_jac=True):
        self.sym = expr
        self.com_jac = com_jac
    
    def cal_jac(self, jac_var):
        return sym.Matrix([self.sym]).jacobian(jac_var)
            
    def get_opt_prob(self, passive_var):
        opt_prob = self.sym.copy()
        for var in passive_var:
            opt_prob = opt_prob.subs(var, var.value)
        return opt_prob
    
    def __call__(self,**kwargs):
        if not hasattr(self,"fun"):
            self.fun = sym.lambdify(list(kwargs.keys()),self.sym)
        return self.fun(**kwargs)
    
    def get_jac(self,obj_opt_self):
        if self.com_jac:
            jac_expr = self.cal_jac(obj_opt_self.get_active_var())
            return sym.lambdify(obj_opt_self.opt_var,jac_expr)
    
class expression_constraint(expression):
    def __init__(self, expr, con_type=">", com_jac=True, active=True):
        self.active = active
        self.con_type = con_type
        super().__init__(expr,com_jac)
        
    def set_passive(self):
        self.active = False
        
    def set_active(self):
        self.active = True
    
        
