# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import math
from util import manhattanDistance
from game import Directions
import random
import util
from math import sqrt, log

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def search(self, state, depth, agent):
            if depth == (self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent == 0:
                return max(search(self, state.generateSuccessor(agent, action), depth, agent+1) for action in state.getLegalActions(agent))
            else:
                agent0 = agent
                agent += 1
                if agent == state.getNumAgents():
                    agent = 0
                    depth += 1
                return min(search(self, state.generateSuccessor(agent0, action), depth, agent) for action in state.getLegalActions(agent0))
        legal = gameState.getLegalActions(0)
        ans = -float("inf")
        for action in legal:
            s = search(self, gameState.generateSuccessor(0, action), 0, 1)
            if s > ans:
                ans = s
                ans_action = action
        return ans_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    def search(self, state, depth, agent, a, b):
        if depth == (self.depth) or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent == 0:
            v = -float("inf")
            for action in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, action)
                v = max(v, self.search(next_state, depth, 1, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v
        else:
            agent0 = agent
            agent += 1
            if agent == state.getNumAgents():
                agent = 0
                depth += 1
            v = float("inf")
            for action in state.getLegalActions(agent0):
                next_state = state.generateSuccessor(agent0, action)
                v = min(v, self.search(next_state, depth, agent, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

    def getAction(self, gameState):
        legal = gameState.getLegalActions(0)
        ans, alpha = -float("inf"), -float("inf")
        for action in legal:
            next_state = gameState.generateSuccessor(0, action)
            s = self.search(next_state, 0, 1, alpha, float("inf"))
            alpha = max(alpha, s)
            if s > ans:
                ans = s
                ans_action = action
        return ans_action


class MCTSAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        for action in actions:
            state = gameState.generateSuccessor(0,action)
            if state.isWin():
                return action
        class Node:
            def __init__(self, data):
                self.children = {}
                self.parent = None
                self.visits = 0
                self.score = 0
                self.state = data[0]
                self.agent = data[1]
            def is_terminal(self):
                return self.state.isWin() or self.state.isLose()
        
        numAgent = gameState.getNumAgents()
        C = math.sqrt(2)
        root = Node([gameState,0])
        def next_agent(index):return (index+1)%numAgent
        def next_action(node):
            state = node.state
            agent = node.agent
            if agent == 0:
                actions = state.getLegalActions(0)
                ans_action, Min_dis = None,float("inf")
                for action in actions:
                    child = node.children.get(action,None)
                    if child == None:
                        new_state = state.generateSuccessor(agent,action)
                        if new_state.isWin() or new_state.isLose():
                            continue
                        pacman_position = new_state.getPacmanPosition()
                        foodlist = state.getFood().asList()
                        dis = min([manhattanDistance(food ,pacman_position) for food in foodlist])
                        if dis < Min_dis:
                            Min_dis = dis
                            ans_action = action
                while ans_action == None or ans_action == "Stop":
                    ans_action = random.choice(actions)
                random_prob = 0.17
                if random.random() < random_prob:
                    ans_action = random.choice(actions)
                return ans_action
            else:
                actions = state.getLegalActions(agent)
                for action in actions:
                    child = node.children.get(action,None)
                    if child == None:
                        return action
                return random.choice(actions)
        def Expansion(node):
            if node.is_terminal():
                return node
            state = node.state
            agent = node.agent
            action = next_action(node)
            new_state = state.generateSuccessor(agent,action)
            new_agent = next_agent(agent)
            new_node = Node([new_state,new_agent])
            new_node.parent = node
            node.children[action] = new_node
            return new_node

        def max_ucb(node):
            best_child = None
            ucb_value = -float("inf")
            actions = node.state.getLegalActions(node.agent)
            for action in actions:
                child = node.children.get(action,None)
                if child == None:
                    return action
                ucb = child.score/child.visits + C * sqrt(log(node.visits)/child.visits)
                if ucb > ucb_value:
                    ucb_value = ucb
                    best_child = child
            return best_child
        def Selection(node):
            while not node.is_terminal():
                action  = max_ucb(node)
                child = node.children.get(action,None)
                if child == None:
                    return node
                node = child
            return node
        def Simulation(node):
            depth = 1
            while not node.is_terminal() and depth <= 0:
                node = Expansion(node)
                depth += 1
            return node
        from math import sqrt, log

        def mcts_eval(state):
            if state.isWin():
                return 1.0
            if state.isLose():
                return -10.0
            food = state.getFood()
            capsules = state.getCapsules()
            ghosts = state.getGhostStates()
            pacman = state.getPacmanPosition()
            scared_time = [ghost.scaredTimer for ghost in ghosts]

            food_distances = [manhattanDistance(pacman, food_pos) for food_pos in food.asList()]
            capsule_distances = [manhattanDistance(pacman, capsule_pos) for capsule_pos in capsules]
            ghost_distances = [manhattanDistance(pacman, ghost.getPosition()) for ghost in ghosts]

            num_food = len(food.asList())
            num_capsules = len(capsules)
            num_ghosts = len(ghosts)

            score = state.getScore()

            if num_food == 0:
                return 1.0

            if any([scared_time[i] == 0 and ghost_distances[i] < 2 for i in range(num_ghosts)]):
                return -10.0

            value  = 3 if numAgent == 2 else 0
            food_score = sum([1.0 / (distance + 1) for distance in food_distances])
            capsule_score = sum([1.0 / (distance + 1) for distance in capsule_distances])
            ghost_score = sum([-1.0 / (distance + 1) / (distance+1) if scared_time[i] == 0 else value / (distance + 1) for i, distance in enumerate(ghost_distances)])

            total_score = 3 * food_score / num_food + 1 * capsule_score / (num_capsules+1) + 5 * ghost_score / num_ghosts + score

            return 1.0 / (1.0 + math.exp(-total_score))


        def Back_propagation(node):
            value = mcts_eval(node.state)
            while True:
                node.visits += 1
                if node == root:
                    break
                if node.parent.agent > 0:
                    node.score += 1-value
                else:
                    node.score += value
                node = node.parent
        if numAgent == 3:
            loop = 1001
        # elif gameState.data.layout.width * gameState.data.layout.height == 756:
        #     loop = 456
        else:
            loop = 666
        for _ in range(loop):
            node = Selection(root)
            child = Expansion(node)
            new_node = Simulation(child)
            Back_propagation(new_node)  
        
        ans_action,mx = None,-float("inf")
        for item in root.children.items():
            value = item[1].score
            if value > mx:
                mx = value
                ans_action = item[0]
        return ans_action