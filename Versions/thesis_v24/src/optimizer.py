"""
Path optimizer for finding optimal paths with visibility constraints.
"""
import logging
import time
from itertools import combinations
import networkx as nx
from gurobipy import Model, GRB
from src.utils import get_subtour, log_memory_usage

class PathOptimizer:
    """Optimizes paths with visibility constraints."""
    
    def __init__(self, config):
        """
        Initialize the path optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, G, segments, segment_visibility, edge_visibility, vrf):
        """
        Optimize the path with visibility constraints.
        
        Args:
            G: networkx DiGraph
            segments: List of segments
            segment_visibility: Dictionary mapping segments to edges that can see them
            edge_visibility: Dictionary mapping edges to segments they can see
            vrf: Dictionary of Visibility Ratio Factor (VRF) for each edge
            
        Returns:
            Tuple of (optimization model, selected edges)
        """
        self.logger.info("Setting up optimization model")
        log_memory_usage(self.logger, "Before optimization model creation")
        
        # Create optimization model
        model = Model("VisibilityPathOptimization")
        
        try:
            # Set model parameters
            self._set_model_parameters(model)
            
            # Create edge variables and cost function
            E_vars, cost = self._create_edge_variables(model, G, vrf)
            
            # Set objective
            self._set_objective(model, E_vars, cost)
            
            # Add segment visibility constraints
            self._add_segment_visibility_constraints(model, segments, segment_visibility, E_vars)
            
            # Add flow constraints
            self._add_flow_constraints(model, G, E_vars)
            
            # Add tie point constraints if specified
            self._add_tie_point_constraints(model, G, E_vars)
            
            # Add subtour elimination constraints (original version)
            self._add_subtour_elimination_constraints(model, G, E_vars)
            
            log_memory_usage(self.logger, "After optimization model setup")
            
            # Solve the model with progress callback
            self.logger.info("Solving optimization model with progress monitoring")
            
            # Initialize progress tracking data
            model._gap_tracking = {
                'start_time': time.time(),
                'last_improvement_time': time.time(),
                'best_gap': float('inf'),
                'improvement_threshold': 0.01,  # 1% improvement threshold
                'stagnation_limit': 600,        # 600 seconds without improvement
                'last_gap': float('inf'),
                'target_gap': 0.05              # 5% target gap
            }
            
            # Solve with both callbacks
            model.optimize(lambda model, where: self._optimization_callback(model, where))
            
            # Get the selected edges
            selected_edges = []
            if model.status == GRB.OPTIMAL:
                selected_edges = [edge for edge in E_vars if E_vars[edge].X > 0.5]
                self.logger.info(f"Optimization successful, selected {len(selected_edges)} edges")
            else:
                # For other statuses (time limit, etc.), still get the best solution found
                if model.SolCount > 0:
                    selected_edges = [edge for edge in E_vars if E_vars[edge].X > 0.5]
                    self.logger.warning(f"Optimization stopped with status {model.status}")
                    
                    # Check if we terminated due to gap stagnation
                    if hasattr(model, '_terminated_by_callback') and model._terminated_by_callback:
                        self.logger.info("Optimization terminated due to gap stagnation (no 1% improvement in 600 seconds) with gap below 5%")
                    
                    # Get optimality gap
                    gap = model.MIPGap * 100  # Convert to percentage
                    self.logger.info(f"Solution found with {gap:.2f}% optimality gap")
                else:
                    self.logger.warning(f"Optimization failed with status {model.status}, no solution found")
            
            log_memory_usage(self.logger, "After optimization model solving")
            
            return model, selected_edges
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}", exc_info=True)
            # Make sure to dispose of the model if an exception occurs
            if 'model' in locals():
                try:
                    model.dispose()
                    self.logger.info("Disposed Gurobi model due to exception")
                except:
                    pass
            raise
    
    def _set_model_parameters(self, model):
        """
        Set optimization model parameters.
        
        Args:
            model: Gurobi optimization model
        """
        # Enable lazy constraints for subtour elimination
        model.Params.LazyConstraints = 1
        
        # Set MIPGap to a very small value to prevent automatic termination
        # Our callback will handle the custom termination logic
        model.Params.MIPGap = 0.0001  # 0.01% gap - effectively disabled
        
        # Check if time limit is specified in config
        time_limit = self.config['optimization']['time_limit']
        if time_limit is not None and time_limit > 0:
            model.Params.TimeLimit = time_limit
            self.logger.info(f"Set optimization time limit to {time_limit} seconds")
    
    def _create_edge_variables(self, model, G, vrf):
        """
        Create edge variables and cost function.
        
        Args:
            G: networkx DiGraph
            vrf: Dictionary of Visibility Ratio Factor (VRF) for each edge
            
        Returns:
            Tuple of (edge variables dictionary, cost dictionary)
        """
        # Initialize dictionaries
        E_vars = {}
        cost = {}
        
        # Edge variables and costs for all edges in the graph
        for i, j in G.edges():
            # Create binary variable for the edge
            E_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"edge_{i}_{j}")
            
            # Calculate edge cost
            edge = (i, j)
            edge_weight = G[i][j]['weight']
            
            # Apply VRF if enabled
            if self.config['optimization']['use_vrf_weight']:
                epsilon = self.config['optimization']['epsilon']
                cost[edge] = edge_weight * (1 / (vrf[edge] + epsilon))
            else:
                cost[edge] = edge_weight
        
        return E_vars, cost
    
    def _set_objective(self, model, E_vars, cost):
        """
        Set the optimization objective.
        
        Args:
            model: Gurobi optimization model
            E_vars: Dictionary of edge variables
            cost: Dictionary of edge costs
        """
        # Minimize the total cost of selected edges
        model.setObjective(
            sum(E_vars[edge] * cost[edge] for edge in E_vars),
            GRB.MINIMIZE
        )
    
    def _add_segment_visibility_constraints(self, model, segments, segment_visibility, E_vars):
        """
        Add constraints to ensure all segments are visible.
        
        Args:
            model: Gurobi optimization model
            segments: List of segments
            segment_visibility: Dictionary mapping segments to edges that can see them
            E_vars: Dictionary of edge variables
        """
        # For each segment, at least one edge that can see it must be selected
        for seg_idx, edges in segment_visibility.items():
            if edges:  # If there are edges that can see this segment
                model.addConstr(
                    sum(E_vars[edge] for edge in edges) >= 1,
                    name=f"seg_visibility_{seg_idx}"
                )
            else:
                self.logger.warning(f"Segment {seg_idx} has no visible edges")
    
    def _add_flow_constraints(self, model, G, E_vars):
        """
        Add flow conservation constraints.
        
        Args:
            model: Gurobi optimization model
            G: networkx DiGraph
            E_vars: Dictionary of edge variables
        """
        # For each node, the number of incoming edges must equal the number of outgoing edges
        for node in G.nodes():
            in_edges = [(i, node) for i in G.predecessors(node) if (i, node) in E_vars]
            out_edges = [(node, j) for j in G.successors(node) if (node, j) in E_vars]
            
            model.addConstr(
                sum(E_vars[edge] for edge in in_edges) == sum(E_vars[edge] for edge in out_edges),
                name=f"flow_{node}"
            )
    
    def _add_tie_point_constraints(self, model, G, E_vars):
        """
        Add constraints for tie points.
    
        Args:
         model: Gurobi optimization model
            G: networkx DiGraph
            E_vars: Dictionary of edge variables
        """
        # Tie points are nodes that the path must pass through multiple times
        tie_points = self.config['optimization']['tie_points']
    
        if not tie_points:
            self.logger.info("No tie points specified in config")
            return
        
        self.logger.info(f"Adding constraints for tie points: {tie_points}")
    
        for node in tie_points:
            if node not in G.nodes():
                self.logger.warning(f"Tie point {node} is not in the graph - skipping")
                continue
        
            in_edges = [(i, node) for i in G.predecessors(node) if (i, node) in E_vars]
            out_edges = [(node, j) for j in G.successors(node) if (node, j) in E_vars]
        
            if not in_edges or not out_edges:
                self.logger.warning(f"Tie point {node} doesn't have enough edges - skipping")
                continue
            
            # Ensure at least 2 incoming and 2 outgoing edges
            model.addConstr(
                sum(E_vars[edge] for edge in in_edges) >= 2,
                name=f"tiepoint_in_{node}"
            )
        
            model.addConstr(
                sum(E_vars[edge] for edge in out_edges) >= 2,
                name=f"tiepoint_out_{node}"
            )
        
            self.logger.info(f"Added tie point constraints for node {node}")
    
    def _add_subtour_elimination_constraints(self, model, G, E_vars):
        """
        Add constraints to eliminate subtours.
        
        Args:
            model: Gurobi optimization model
            G: networkx DiGraph
            E_vars: Dictionary of edge variables
        """
        # Original implementation: Pre-elimination for small subtours (3 nodes)
        self.logger.info("Using original subtour elimination constraints")
        
        constraint_count = 0
        for subtour in combinations(G.nodes(), 3):
            # Sum of edges within the subtour
            edges_in_subtour = [
                (i, j) for i in subtour for j in subtour
                if i != j and (i, j) in E_vars
            ]
            
            if edges_in_subtour:
                # A valid tour can have at most 2 edges among any 3 nodes
                model.addConstr(
                    sum(E_vars[edge] for edge in edges_in_subtour) <= 2,
                    name=f"subtour3_{subtour}"
                )
                constraint_count += 1
        
        self.logger.info(f"Added {constraint_count} subtour elimination constraints")
        
        # Store variables for callback
        model._vars = E_vars
        model._nodes = list(G.nodes())
    
    def _optimization_callback(self, model, where):
        """
        Combined callback for progress tracking and subtour elimination.
        
        Args:
            model: Gurobi optimization model
            where: Callback location
        """
        # Subtour elimination part
        if where == GRB.Callback.MIPSOL:
            # Get current solution
            sol = model.cbGetSolution(model._vars)
            selected = [(i, j) for (i, j) in model._vars if sol[i, j] > 0.5]
            
            # Find used nodes
            used_nodes = set()
            for i, j in selected:
                used_nodes.add(i)
                used_nodes.add(j)
            used_nodes = list(used_nodes)
            
            # Check for subtours
            subtour = get_subtour(used_nodes, selected)
            
            if subtour is not None:
                # Add lazy constraint: sum of edges in subtour <= |subtour| - 1
                expr = 0
                for i in subtour:
                    for j in subtour:
                        if (i, j) in model._vars:
                            expr += model._vars[(i, j)]
                
                model.cbLazy(expr <= len(subtour) - 1)
        
        # Progress tracking part
        elif where == GRB.Callback.MIP:
            # Only check if we have at least one solution
            if model.cbGet(GRB.Callback.MIP_SOLCNT) > 0:
                # Get current gap
                current_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                current_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                
                if current_bound != 0:  # Avoid division by zero
                    current_gap = abs((current_best - current_bound) / current_bound)
                else:
                    current_gap = float('inf')
                
                # Get current time
                current_time = time.time()
                elapsed = current_time - model._gap_tracking['start_time']
                
                # Display progress every 60 seconds
                if (current_time - model._gap_tracking.get('last_log_time', 0)) > 60:
                    self.logger.info(f"Optimization progress: {current_gap*100:.2f}% gap after {elapsed:.1f} seconds")
                    model._gap_tracking['last_log_time'] = current_time
                
                # If we found a better gap, update tracking
                if current_gap < model._gap_tracking['best_gap']:
                    # Calculate improvement
                    improvement = model._gap_tracking['best_gap'] - current_gap
                    
                    # If improvement is significant (≥ 1%), update last improvement time
                    if improvement >= model._gap_tracking['improvement_threshold']:
                        model._gap_tracking['last_improvement_time'] = current_time
                        self.logger.info(f"Gap improved by {improvement*100:.2f}% to {current_gap*100:.2f}%")
                    
                    # Update best gap
                    model._gap_tracking['best_gap'] = current_gap
                
                # Check termination criteria: below target gap (5%) AND no 1% improvement for 600 seconds
                stagnation_time = current_time - model._gap_tracking['last_improvement_time']
                target_gap = model._gap_tracking['target_gap']
                
                if (current_gap <= target_gap) and (stagnation_time > model._gap_tracking['stagnation_limit']):
                    self.logger.info(f"Terminating: Gap is {current_gap*100:.2f}% (below {target_gap*100}%) with no 1% improvement in {stagnation_time:.1f} seconds")
                    model._terminated_by_callback = True
                    model.terminate()
                
                # Save current gap for next iteration
                model._gap_tracking['last_gap'] = current_gap