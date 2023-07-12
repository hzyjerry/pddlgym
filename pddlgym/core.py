"""Implements PDDLEnv, a gym.Env parameterized by PDDL.

One PDDLEnv corresponds to one PDDL domain. Each episode corresponds to
one one PDDL problem; calling env.reset() sets up a new problem.

Observations are namedtuples with attributes `literals`, `objects`, `goal`.
Actions are single ground literals (not operators -- see README).

The debug_info returned by reset and step contains the domain PDDL file
and current problem PDDL file to facilitate interaction with a planner.

Usage example:
>>> env = PDDLEnv("pddl/sokoban.pddl", "pddl/sokoban")
>>> obs, debug_info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, done, debug_info = env.step(action)
"""
from pddlgym.parser import PDDLDomainParser, PDDLProblemParser, PDDLParser
from pddlgym.inference import find_satisfying_assignments, check_goal
from pddlgym.structs import ground_literal, Literal, State, ProbabilisticEffect, LiteralConjunction, NoChange
from pddlgym.spaces import LiteralSpace, LiteralSetSpace, LiteralActionSpace

import copy
import functools
import glob
import os
from itertools import product

import gym
import time
import numpy as np



class InvalidAction(Exception):
    """See PDDLEnv docstring"""
    pass

def get_successor_state(state, action, domain, raise_error_on_invalid_action=False,
                        inference_mode="infer", require_unique_assignment=True, get_all_transitions=False, return_probs=False):
    """
    Compute successor state(s) using operators in the domain

    Parameters
    ----------
    state : State
    action : Literal
    domain : PDDLDomain
    raise_error_on_invalid_action : bool
    inference_mode : "csp" or "prolog" or "infer"
    require_unique_assignment : bool
    get_all_transitions : bool
        If true, this function returns all possible successor states in the case that probabilistic effects exist in the domain.

    Returns
    -------
    next_state : State
    """
    selected_operator, assignment = _select_operator(state, action, domain,
        inference_mode=inference_mode,
        require_unique_assignment=require_unique_assignment)

    # A ground operator was found; execute the ground effects
    if assignment is not None:
        # Get operator effects
        if isinstance(selected_operator.effects, LiteralConjunction):
            effects = selected_operator.effects.literals
        else:
            assert isinstance(selected_operator.effects, Literal)
            effects = [selected_operator.effects]

        state = _apply_effects(
            state,
            effects,
            assignment,
            get_all_transitions,
            return_probs=return_probs,
        )

    # No operator was found
    elif raise_error_on_invalid_action:
        raise InvalidAction(f"called get_successor_state with invalid action '{action}' for given state")

    return state


def get_successor_states(state, action, domain, raise_error_on_invalid_action=False,
                         inference_mode="infer", require_unique_assignment=True, return_probs=False):
    return get_successor_state(state, action, domain, raise_error_on_invalid_action, inference_mode, require_unique_assignment, get_all_transitions=True, return_probs=return_probs)


def _select_operator(state, action, domain, inference_mode="infer",
                     require_unique_assignment=True):
    """
    Helper for successor generation
    """
    if inference_mode == "infer":
        inference_mode = "csp" if _check_domain_for_strips(domain) else "prolog"

    single_agent = type(action) == Literal

    # Single agent
    if single_agent:
        if domain.operators_as_actions:
            # There should be only one possible operator if actions are operators
            possible_operators = set()
            for name, operator in domain.operators.items():
                if name.lower() == action.predicate.name.lower():
                    assert len(possible_operators) == 0
                    possible_operators.add(operator)
        else:
            # Possibly multiple operators per action
            possible_operators = set(domain.operators.values())

        # Knowledge base: literals in the state + action taken
        kb = set(state.literals) | {action}

    else: # Multiagent
        possible_operators = set(domain.operators.values())
        # Knowledge base: literals in the state + action taken
        kb = set(state.literals) | set(action.literals)

    selected_operator = None
    assignment = None
    for operator in possible_operators:
        t1 = time.time()

        if isinstance(operator.preconds, Literal):
            conds = [operator.preconds]
        else:
            conds = operator.preconds.literals
        # Necessary for binding the operator arguments to the variables
        if type(action) == Literal and domain.operators_as_actions:
            conds = [action.predicate(*operator.params)] + conds

        # Check whether action is in the preconditions
        def _filter_by_action_predicate(inputs, predicate):
            action
            operator
            action_literal = None
            for lit in conds:
                if lit.predicate == predicate:
                    action_literal = lit
                    break
            action_variables = action_literal.variables
            # variable_sort_fn = lambda v : (not v in action_variables, v)
            def variable_sort_fn(v):
                return (not v in action_variables, v)
            outputs = find_satisfying_assignments(inputs, conds,
                variable_sort_fn=variable_sort_fn,
                type_to_parent_types=domain.type_to_parent_types,
                max_assignment_count=999,
                constants=domain.constants,
                mode=inference_mode)
            return outputs

        # Check whether action is in the preconditions
        def _filter_by_multi_action_predicates(inputs, literals):
            action
            operator
            action_literal = None
            all_action_variables = []
            given_predicates = [l.predicate for l in literals]
            for lit in conds:
                if lit.predicate in given_predicates:
                    action_literal = lit
                    all_action_variables += lit.variables
            def variable_sort_fn(v):
                return (not v in all_action_variables, v)
            assigned_variables = [] # Given assignments, help prune search
            for l in literals:
                assigned_variables += l.variables
            outputs = find_satisfying_assignments(inputs, conds,
                variable_sort_fn=variable_sort_fn,
                type_to_parent_types=domain.type_to_parent_types,
                max_assignment_count=999,
                constants=domain.constants,
                mode=inference_mode,
                assigned_variables=assigned_variables
            )
            return outputs

        # Handle literal conjunctions
        if single_agent:
            assignments = _filter_by_action_predicate(kb, action.predicate)
        else:
            if len(action.literals) == 1:
                return _filter_by_action_predicate(kb, action.literals[0].predicate)

            # There are multiple literals, we need to find the 1 assignment(s) jointly selected by all literals
            all_lit_assignments = []
            assignments = _filter_by_multi_action_predicates(kb, action.literals)

        # print(f"Step 2.1 time {time.time() - t1:.3f}")
        t1 = time.time()

        # Ensure that assignment satisfy literals
        def _filter_by_action_variables(assignments, literals):
            out_assignments = []
            for assign in assignments:
                satisfied = True
                for lit in literals:
                    # TODO: currently only move(xxx) or is-plauer(xxx), what if length > 1
                    for v in lit.variables:
                        if v not in assign.values():
                            satisfied = False
                if satisfied:
                    out_assignments.append(assign)
            return out_assignments

        num_assignments = len(assignments)
        if single_agent: # action is a literal
            assignments = _filter_by_action_variables(assignments, [action])
        else:
            assignments = _filter_by_action_variables(assignments, action.literals)
        num_assignments = len(assignments)

        if num_assignments > 0:
            if require_unique_assignment:
                assert num_assignments == 1, "Nondeterministic envs not supported"
            selected_operator = operator
            assignment = assignments[0]
            break

        # print(f"Step 2.2 time {time.time() - t1:.3f}")
        t1 = time.time()

    return selected_operator, assignment

def _check_all_action_variables(assignments, action):
    """
    In multi-agent case, ensure that we only assign the agent specified by the action
    """
    out_assignments = []
    for assign in assignments:
        select = True
        for act_var in action.variables:
            at_least_one_satisfy = False
            for assign_key, assign_var in assign.items():
                if assign_var.var_type == act_var.var_type and assign_var.name == act_var.name:
                    at_least_one_satisfy = True
            select = select and at_least_one_satisfy
        if select:
            out_assignments.append(assign)
    return out_assignments


def _check_domain_for_strips(domain):
    """
    Check whether all operators in a domain are STRIPS
    """
    for operator in domain.operators.values():
        if not _check_struct_for_strips(operator.preconds):
            return False
    return True

def _check_struct_for_strips(struct):
    """
    Helper for _check_domain_for_strips
    """
    if isinstance(struct, Literal):
        return True
    if isinstance(struct, LiteralConjunction):
        return all(_check_struct_for_strips(l) for l in struct.literals)
    return False


def _compute_new_state_from_lifted_effects(lifted_effects, assignments, new_literals):
    for lifted_effect in lifted_effects:
        if lifted_effect == NoChange():
            continue
        effect = ground_literal(lifted_effect, assignments)
        # Negative effect
        if effect.is_anti:
            literal = effect.inverted_anti
            if literal in new_literals:
                new_literals.remove(literal)
    for lifted_effect in lifted_effects:
        if lifted_effect == NoChange():
            continue
        effect = ground_literal(lifted_effect, assignments)
        if not effect.is_anti:
            new_literals.add(effect)
    return new_literals

def _apply_effects(state, lifted_effects, assignments, get_all_transitions=False,
                   return_probs=False):
    """
    Update a state given lifted operator effects and
    assignments of variables to objects.

    Parameters
    ----------
    state : State
        The state on which the effects are applied.
    lifted_effects : { Literal }
    assignments : { TypedEntity : TypedEntity }
        Maps variables to objects.
    get_all_transitions : bool
        If true, this function returns all possible successor states in the case that probabilistic effects exist in the domain.
    """
    new_literals = set(state.literals)
    determinized_lifted_effects = []
    # Handle probabilistic effects.

    # Each element of this list contain
    #   a pair of outcomes from a probabilistic effect
    probabilistic_lifted_effects = []
    for lifted_effect in lifted_effects:
        if isinstance(lifted_effect, ProbabilisticEffect):
            effect_outcomes = lifted_effect.literals
            probas = dict(zip(lifted_effect.literals,
                              lifted_effect.probabilities))
            cur_probabilistic_lifted_effects = []


            if get_all_transitions:
                lifted_effects_list = cur_probabilistic_lifted_effects
            else:
                lifted_effects_list = determinized_lifted_effects
            sampled_effect = lifted_effect.sample()


            # If get_all_transitions == False, create list with sampled state only
            # Otherwise, populate it with possible outcomes
            effects_to_process = [
                sampled_effect] if not get_all_transitions else effect_outcomes

            for chosen_effect in effects_to_process:
                if isinstance(chosen_effect, LiteralConjunction):
                    for lit in chosen_effect.literals:
                        lifted_effects_list.append(lit)
                        lit.proba = probas[chosen_effect]
                else:
                    lifted_effects_list.append(chosen_effect)
                    chosen_effect.proba = probas[chosen_effect]

            if get_all_transitions:
                probabilistic_lifted_effects.append(
                    cur_probabilistic_lifted_effects)
        else:
            determinized_lifted_effects.append(lifted_effect)

    states = []
    if not get_all_transitions:
        new_literals = _compute_new_state_from_lifted_effects(determinized_lifted_effects, assignments, new_literals)

        return state.with_literals(new_literals)

    # else - get all possible transitions

    # Construct combinations of probabilistic effects
    probabilistic_effects_combinations = list(
        product(*probabilistic_lifted_effects))

    states_to_probs = {}
    for prob_efs_combination in probabilistic_effects_combinations:
        total_proba = np.prod([lit.proba for lit in prob_efs_combination])
        if total_proba == 0:
            continue
        new_prob_literals = set(state.literals)
        new_determinized_lifted_effects = determinized_lifted_effects + \
            list(prob_efs_combination)
        new_prob_literals = _compute_new_state_from_lifted_effects(new_determinized_lifted_effects, assignments, new_prob_literals)

        new_state = state.with_literals(new_prob_literals)
        if new_state in states_to_probs:
            # If there are multiple ways of reaching next state,
            #   then these probabilities have to be summed
            states_to_probs[new_state] += total_proba
        else:
            states_to_probs[new_state] = total_proba
        states.append(new_state)
    if return_probs:
        return states_to_probs
    # convert list of states to set
    return frozenset(states)


class PDDLEnv(gym.Env):
    """
    Parameters
    ----------
    domain_file : str
        Path to a PDDL domain file.
    problem_dir : str
        Path to a directory of PDDL problem files.
    render : fn or None
        An optional render function (obs -> img).
    seed : int
        Random seed used to sample new problems upon reset.
    raise_error_on_invalid_action : bool
        When an action is taken for which no operator's
        preconditions holds, raise InvalidAction() if True;
        otherwise silently make no changes to the state.
    operators_as_actions : bool
        If True, the PDDL operators are treated as the actions.
        Otherwise, actions must be specified separately in the PDDL file.
    dynamic_action_space : bool
        Let self.action_space dynamically change on each iteration to
        include only valid actions (must match operator preconditions).
    """
    def __init__(self, domain_file, problem_dir, render=None, seed=0,
                 handle_derived_literals=False, # Modified
                 num_players=1,
                 raise_error_on_invalid_action=False,
                 operators_as_actions=True, # Modified
                 dynamic_action_space=False,
                 max_timestep=-1):
        self._state = None
        self._domain_file = domain_file
        self._problem_dir = problem_dir
        self._render = render
        self.seed(seed)
        self._raise_error_on_invalid_action = raise_error_on_invalid_action
        self.operators_as_actions = operators_as_actions
        self.handle_derived_literals = handle_derived_literals

        # Number of players:
        self._num_players = num_players

        # Set by self.fix_problem_index
        self._problem_index_fixed = False

        self._problem_idx = None
        self._max_timestep = max_timestep

        # Parse the PDDL files
        self.domain, self.problems = self.load_pddl(domain_file, problem_dir,
            operators_as_actions=self.operators_as_actions)

        # Determine if the domain is STRIPS
        self._domain_is_strips = _check_domain_for_strips(self.domain)
        self._inference_mode = "csp" if self._domain_is_strips else "prolog"

        # Initialize action space with problem-independent components
        actions = list(self.domain.actions)
        self.action_predicates = [self.domain.predicates[a] for a in actions]
        self._dynamic_action_space = dynamic_action_space
        if dynamic_action_space:
            if self.domain.operators_as_actions and self._domain_is_strips:
                self._action_space = LiteralActionSpace(
                    self.domain, self.action_predicates,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types)
            else:
                self._action_space = LiteralSpace(
                    self.action_predicates, lit_valid_test=self._action_valid_test,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types)

        else:
            self._action_space = LiteralSpace(self.action_predicates,
                type_to_parent_types=self.domain.type_to_parent_types)

        # Initialize observation space with problem-independent components
        self._observation_space = LiteralSetSpace(
            set(self.domain.predicates.values()) - set(self.action_predicates),
            type_hierarchy=self.domain.type_hierarchy,
            type_to_parent_types=self.domain.type_to_parent_types)

    @staticmethod
    def load_pddl(domain_file, problem_dir, operators_as_actions=False):
        """
        Parse domain and problem PDDL files.

        Parameters
        ----------
        domain_file : str
            Path to a PDDL domain file.
        problem_dir : str
            Path to a directory of PDDL problem files.
        operators_as_actions : bool
            See class docstirng.

        Returns
        -------
        domain : PDDLDomainParser
        problems : [ PDDLProblemParser ]
        """
        domain = PDDLDomainParser(domain_file,
            expect_action_preds=(not operators_as_actions),
            operators_as_actions=operators_as_actions)
        problems = []
        if problem_dir is not None:
            problem_files = [f for f in glob.glob(os.path.join(problem_dir, "*.pddl"))]
            for problem_file in sorted(problem_files):
                problem = PDDLProblemParser(problem_file, domain.domain_name,
                    domain.types, domain.predicates, domain.actions, domain.constants)
                problems.append(problem)
        return domain, problems

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def problem_idx(self):
        return self._problem_idx

    @property
    def problem(self):
        return self._problem

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def seed(self, seed):
        self._seed = seed
        self.rng = np.random.RandomState(seed)

    def fix_problem_index(self, problem_idx):
        """
        Fix the PDDL problem used when reset is called.

        Useful for reproducible testing.

        The order of PDDL problems is determined by the names
        of their files. See PDDLEnv.load_pddl.

        Parameters
        ----------
        problem_idx : int
        """
        self._problem_idx = problem_idx
        self._problem_index_fixed = True

    def fix_problem(self, problem_file):
        domain = self.domain
        problem = PDDLProblemParser(problem_file, domain.domain_name, domain.types,
            domain.predicates, domain.actions, domain.constants)
        self.problems = [problem]
        self.fix_problem_index(0)

    def fix_problem_str(self, problem_str):
        domain = self.domain
        problem = PDDLProblemParser(None, domain.domain_name, domain.types,
            domain.predicates, domain.actions, domain.constants, problem_str=problem_str)
        self.problems = [problem]
        self.fix_problem_index(0)

    def reset(self):
        """
        Set up a new PDDL problem and start a new episode.

        Note that the PDDL files are included in debug_info.

        Returns
        -------
        obs : { Literal }
            The set of active predicates.
        debug_info : dict
            See self._get_debug_info()
        """
        if not self._problem_index_fixed:
            self._problem_idx = self.rng.choice(len(self.problems))
        self._problem = self.problems[self._problem_idx]

        initial_state = State(frozenset(self._problem.initial_state),
                              frozenset(self._problem.objects),
                              self._problem.goal)
        if self.handle_derived_literals:
            initial_state = self._handle_derived_literals(initial_state)
        self.set_state(initial_state)

        self._goal = self._problem.goal
        debug_info = self._get_debug_info()

        self._timestep = 0
        self._action_space.reset_initial_state(initial_state)

        return self.get_state(), debug_info

    def _get_debug_info(self):
        """
        Contains the problem file and domain file
        for interaction with a planner.
        """
        info = {'problem_file' : self._problem.problem_fname,
                'domain_file' : self.domain.domain_fname }
        return info

    def step(self, action):
        """
        Execute an action and update the state.

        Tries to find a ground operator for which the
        preconditions hold when this action is taken. If none
        exist, optionally raises InvalidAction. If multiple
        exist, raises an AssertionError, since we assume
        deterministic environments only. Once the operator
        is found, the ground effects are executed to update
        the state.

        Parameters
        ----------
        action : Literal

        Returns
        -------
        state : State
            The set of active predicates.
        reward : float
            1 if the goal is reached and 0 otherwise.
        done : bool
            True if the goal is reached.
        debug_info : dict
            See self._get_debug_info.
        """
        t1 = time.time()
        reward, done, debug_info = 0, False, None
        for acs in action:
            state, rew, d, debug_info = self.sample_transition(acs)
            self.set_state(state)
            # i.e. goal -> AND[at-goal(stone-01:thing,pos-2-4:location), at-goal(stone-02:thing,pos-4-2:location)]
            reward = min(reward + rew, 1)
            done = done or d
        self._timestep += 1
        if self._max_timestep > 0:
            done = done or self._timestep >= self._max_timestep
        return state, reward, done, debug_info

    def _get_new_state_info(self, state):
        if self.handle_derived_literals:
            state = self._handle_derived_literals(state)

        done = self._is_goal_reached(state)

        reward = self.extrinsic_reward(state, done)
        debug_info = self._get_debug_info()

        return state, reward, done, debug_info

    def sample_transition(self, action):
        if action is None:
            return self._get_new_state_info(self._state)
        else:
            state = self._get_successor_state(self._state, action, self.domain,
                                              inference_mode=self._inference_mode,
                                              raise_error_on_invalid_action=self._raise_error_on_invalid_action)
            return self._get_new_state_info(state)

    def _get_successor_state(self, *args, **kwargs):
        """Separated out to allow for overrides in subclasses
        """
        return get_successor_state(*args, **kwargs)

    def _get_successor_states(self, *args, **kwargs):
        """Separated out to allow for overrides in subclasses
        """
        return get_successor_states(*args, **kwargs)

    def get_all_possible_transitions(self, action, return_probs=False):
        assert self.domain.is_probabilistic
        states = self._get_successor_states(self._state, action, self.domain,
                                            inference_mode=self._inference_mode,
                                            raise_error_on_invalid_action=self._raise_error_on_invalid_action, return_probs=return_probs)
        if return_probs:
            return [(self._get_new_state_info(state), prob) for state, prob in states.items()]

        return [self._get_new_state_info(state) for state in states]

    def extrinsic_reward(self, state, done):
        if done:
            reward = 1.
        else:
            reward = 0.

        return reward

    def _is_goal_reached(self, state):
        """
        Check if the terminal condition is met, i.e., the goal is reached.
        """
        return check_goal(state, self._goal)

    def _action_valid_test(self, state, action):
        _, assignment = _select_operator(state, action, self.domain,
            inference_mode=self._inference_mode)
        return assignment is not None

    def render(self, *args, **kwargs):
        if self._render:
            return self._render(self._state.literals, *args, goal=self._problem.goal, **kwargs)

    def _handle_derived_literals(self, state):
        # first remove any old derived literals since they're outdated

        to_remove = set()
        for lit in state.literals:
            if lit.predicate.is_derived:
                to_remove.add(lit)
        state = state.with_literals(state.literals - to_remove)

        # add negative basic literals for checking derived predicates
        state_literals = state.literals
        all_ground_literals = self._observation_space.all_ground_literals(state)
        for lit in all_ground_literals:
            if not lit.predicate.is_derived and lit not in state_literals:
                state_literals = {lit.negative} | state_literals


        while True:  # loop, because derived predicates can be recursive
            new_derived_literals = set()
            for pred in self.domain.predicates.values():
                if not pred.is_derived:
                    continue
                assignments = find_satisfying_assignments(
                    state_literals, pred.body,
                    type_to_parent_types=self.domain.type_to_parent_types,
                    constants=self.domain.constants,
                    mode="prolog",
                    max_assignment_count=99999)
                for assignment in assignments:
                    objects = [assignment[param_type(param_name)]
                               for param_name, param_type in zip(pred.param_names, pred.var_types)]
                    derived_literal = pred(*objects)
                    if derived_literal not in state.literals:
                        new_derived_literals.add(derived_literal)
            if new_derived_literals:
                # update state_literals for recursive checking
                state_literals = state_literals | new_derived_literals
                # save derived literals in state
                state = state.with_literals(state.literals | new_derived_literals)
            else:  # terminate
                break


        return state
