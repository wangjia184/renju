use crate::*;

use core::cell::RefCell;
use std::{
    collections::HashMap,
    rc::{Rc, Weak},
};

pub struct TreeNode {
    action: Option<(usize, usize)>,
    parent: Option<Weak<RefCell<TreeNode>>>, // None if this is a root node
    children: HashMap<(usize /*row*/, usize /*col*/), Rc<RefCell<TreeNode>>>,
    current: Option<Weak<RefCell<TreeNode>>>,
    stones: u32,      // number of stones on board without couting this node
    visit_times: u32, // number of visited times
    probability: f32, // prior probability from policy network

    // UCB is based on the principle of “optimism in the fact of uncertainty”,
    // which basically means if you don’t know which action is best
    // then choose the one that currently looks to be the best.
    // UCB = q + u

    // The first half of the equation will do exactly that:
    // the action that currently has the highest estimated reward will be the chosen action.
    q: f32, // the exploitation part of the equation. average of evaluations of all leaves
}

impl TreeNode {
    pub fn new(probability: f32) -> Rc<RefCell<TreeNode>> {
        let node = TreeNode {
            action: None,
            parent: None,
            children: HashMap::new(),
            current: None,
            stones: 0,
            visit_times: 0,
            probability: probability,
            q: 0f32,
            //u: 0f32,
        };
        let instance = Rc::new(RefCell::new(node));
        instance.borrow_mut().current = Some(Rc::downgrade(&instance));
        instance
    }

    pub fn new_child(pos: (usize, usize), probability: f32) -> Rc<RefCell<TreeNode>> {
        let node = TreeNode {
            action: Some(pos),
            parent: None,
            children: HashMap::new(),
            current: None,
            stones: 0,
            visit_times: 0,
            probability: probability,
            q: 0f32,
            //u: 0f32,
        };
        let instance = Rc::new(RefCell::new(node));
        instance.borrow_mut().current = Some(Rc::downgrade(&instance));
        instance
    }

    // Create a new child
    pub fn create_child(
        self: &mut TreeNode,
        pos: (usize, usize),
        probability: f32,
    ) -> Rc<RefCell<TreeNode>> {
        let instance = TreeNode::new_child(pos, probability);
        self.append_child(pos, instance.clone());
        instance
    }

    // Append a child
    fn append_child(self: &mut TreeNode, pos: (usize, usize), child: Rc<RefCell<TreeNode>>) {
        assert!(self.current.is_some());
        // link child node to this one
        {
            let mut node = child.borrow_mut();
            node.parent = Some(self.current.as_ref().unwrap().clone());
            node.stones = self.stones + 1;
        }
        child.borrow_mut().parent = Some(self.current.as_ref().unwrap().clone());
        self.children.insert(pos, child);
    }

    #[allow(dead_code)]
    // Determine if this is a leaf node
    pub fn is_leaf(self: &TreeNode) -> bool {
        self.children.len() == 0
    }

    #[allow(dead_code)]
    // Determine if this is root node
    pub fn is_root(self: &TreeNode) -> bool {
        self.parent.is_none()
    }

    // Update node values from leaf evaluation.
    // leaf_value: the value of subtree evaluation from the current player's perspective.
    pub fn update(self: &mut TreeNode, leaf_value: f32) {
        self.visit_times += 1;
        // q is the avarage score(evaluabl) in visit_times, initially q is zero
        // q = sum(value) / n
        // Progressive update : new_q = (value - old_q) / n + old_q
        self.q += 1.0 * (leaf_value - self.q) / (self.visit_times as f32);
    }

    // Like a call to update(), but applied recursively for all ancestors.
    pub fn update_recursive(self: &mut TreeNode, leaf_value: f32) {
        // If it is not root, this node's parent should be updated first.
        if let Some(parent) = &self.parent {
            let parent = parent
                .upgrade()
                .expect("Unable to upgrade weak reference of parent node");
            // Odd and even levels in the tree are from different players (black/white)
            // the leaf node is current turn player. The parent node is opposite player's move
            // hence the value must multiple -1 when traverse from bottom to top
            parent.borrow_mut().update_recursive(-leaf_value);
        }
        self.update(leaf_value);
    }

    // Calculate and return the value for this node.
    // It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
    // c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
    pub fn compute_score(self: &Self, c_puct: f32) -> f32 {
        // The second half of the equation adds exploration,
        // with the degree of exploration being controlled by the hyper-parameter ‘c’.
        // Effectively this part of the equation provides a measure of the uncertainty for the action’s reward estimate.
        // u = visit-count-adjusted prior score
        let u = match &self.parent {
            Some(parent) => {
                let parent_visit_times = parent
                    .upgrade()
                    .expect("Unable to upgrade weak reference of parent node")
                    .as_ref()
                    .borrow()
                    .visit_times;

                c_puct * self.probability * f32::sqrt(parent_visit_times as f32)
                    / (1f32 + self.visit_times as f32)
            }
            None => 0f32,
        };
        return self.q + u;
    }

    /// Select a child with max UCB(Q+U)
    pub fn select(self: &Self, c_puct: f32) -> Option<((usize, usize), Rc<RefCell<TreeNode>>)> {
        let mut max_score = f32::MIN;
        let mut selected_pair: Option<((usize, usize), Rc<RefCell<TreeNode>>)> = None;

        for (pos, child) in &self.children {
            let score = child.borrow().compute_score(c_puct);
            if score > max_score {
                max_score = score;
                selected_pair = Some((*pos, child.clone()));
            }
        }

        selected_pair
    }

    #[allow(dead_code)]
    pub fn get_q(self: &Self) -> f32 {
        self.q
    }
    #[allow(dead_code)]
    pub fn get_action(self: &Self) -> Option<(usize, usize)> {
        self.action
    }
    #[allow(dead_code)]
    pub fn get_child(self: &Self, pos: (usize, usize)) -> Option<Rc<RefCell<TreeNode>>> {
        self.children.get(&pos).and_then(|x| Some(x.clone()))
    }
}

// Monte Carlo tree search
pub struct MonteCarloTree<M>
where
    M: RenjuModel,
{
    model: Rc<RefCell<M>>,
    c_puct: f32,
    root: Rc<RefCell<TreeNode>>,
}

impl<M> MonteCarloTree<M>
where
    M: RenjuModel,
{
    pub fn new(c_puct: f32, model: Rc<RefCell<M>>) -> Self {
        Self {
            c_puct: c_puct,
            root: TreeNode::new(1f32),
            model: model,
        }
    }
    #[allow(dead_code)]
    pub fn get_root(self: &Self) -> Rc<RefCell<TreeNode>> {
        self.root.clone()
    }

    pub fn rollout(self: &mut Self, mut board: RenjuBoard, choices: &Vec<(usize, usize)>) {
        let mut node = self.root.clone();
        assert_eq!(board.get_last_move(), node.borrow().action);

        let mut state: Option<TerminalState> = None;
        if !node.borrow().children.is_empty() {
            loop {
                // Greedily select next move.
                let child = match node.borrow().select(self.c_puct) {
                    None => {
                        break;
                    } // leaf
                    Some((pos, c)) => {
                        state = Some(board.do_move(pos));
                        c
                    }
                };
                node = child;
                if state.as_ref().unwrap().is_over() {
                    break;
                }
            }
        } else {
            if choices.len() > 1 {
                println!("Never explored before");
            }

            // no children, never explored this branch. hence we need use the available moves from outside
            state = Some(TerminalState::AvailableMoves(choices.clone()));
        }

        // score for node
        let evaluation_score: f32;

        assert!(state.is_some());

        match state.unwrap() {
            TerminalState::AvailableMoves(choices) => {
                assert!(!choices.is_empty());

                // Evaluate the leaf using a network
                let state_tensor: [StateTensor; 1] = [board.get_state_tensor()];

                let (prob_matrix, score) = self
                    .model
                    .borrow()
                    .predict(&state_tensor, false)
                    .expect("Failed to predict");

                // black and white are placed in turns
                // if `node` is a black move, then `score` is an evaluation from white's perspective.
                // Since this is a zero-sum game. Positive score means advantages for white, then disadvantages for black.
                // That's why we need negative score because it is from opposite perspective.
                evaluation_score = -score;

                // extend children
                for (row, col) in choices {
                    let probability = prob_matrix[row][col];
                    node.borrow_mut().create_child((row, col), probability);
                }
            }
            TerminalState::Draw => {
                evaluation_score = 0f32;
            }
            TerminalState::BlackWon => {
                if board.is_black_turn() {
                    // node is white move and lost
                    evaluation_score = -1f32;
                } else {
                    // node is black move and won
                    evaluation_score = 1f32;
                }
            }
            TerminalState::WhiteWon => {
                if board.is_black_turn() {
                    // node is white move and won
                    evaluation_score = 1f32;
                } else {
                    // node is black move and lost
                    evaluation_score = -1f32;
                }
            }
        };

        node.borrow_mut().update_recursive(evaluation_score);
    }

    // temperature parameter in (0, 1] controls the level of exploration
    pub fn get_move_probability(
        self: &mut Self,
        temperature: f32,
    ) -> Vec<((usize, usize) /*pos*/, f32 /* probability */)> {
        let mut max_log_visit_times = 0f32;
        // calc the move probabilities based on visit counts at the root node
        let pairs: Vec<_> = self
            .root
            .borrow()
            .children
            .iter()
            .map(|(pos, child)| {
                let log_visit_times = 1f32 / temperature
                    * (1e-10 /*avoid zero*/ + child.borrow().visit_times as f32).ln();
                if log_visit_times > max_log_visit_times {
                    max_log_visit_times = log_visit_times;
                }
                (*pos, log_visit_times)
            })
            .collect();

        // softmax
        let mut sum = 0f32;
        let mut pairs: Vec<_> = pairs
            .into_iter()
            .map(|(pos, log_visit_times)| {
                let prob = (log_visit_times - max_log_visit_times).exp();
                sum += prob;
                (pos, prob)
            })
            .collect();

        pairs.iter_mut().for_each(|(_, prob)| {
            *prob /= sum;
        });

        pairs
    }

    pub fn update_with_position(self: &mut Self, pos: (usize, usize)) {
        let child = {
            let mut r = self.root.borrow_mut();

            r.children
                .remove(&pos)
                .or_else(|| Some(r.create_child(pos, 0f32)))
                .unwrap()
        };

        self.root.borrow_mut().children.clear(); // free all other children
        if let Ok(cell) = Rc::try_unwrap(child) {
            let mut node = cell.into_inner();
            node.parent = None;
            self.root.replace(node);
            self.root.borrow_mut().current = Some(Rc::downgrade(&self.root));
            self.root
                .borrow_mut()
                .children
                .iter_mut()
                .for_each(|(_, child)| {
                    child.borrow_mut().parent = Some(Rc::downgrade(&self.root));
                });
        } else {
            unreachable!("There must be only one strong reference to child node");
        }
    }
}

// for integration test only
#[cfg(test)]
#[allow(dead_code)]
impl TreeNode {
    pub fn get_visit_times(self: &Self) -> u32 {
        self.visit_times
    }
    pub fn get_probability(self: &Self) -> f32 {
        self.probability
    }
    pub fn get_parent(self: &Self) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(ref parent) = self.parent {
            return parent.upgrade();
        }
        None
    }

    pub fn get_children(
        self: &Self,
    ) -> HashMap<(usize /*row*/, usize /*col*/), Rc<RefCell<TreeNode>>> {
        self.children.clone()
    }

    pub fn get_current(self: &Self) -> Rc<RefCell<TreeNode>> {
        if let Some(ref current) = self.current {
            if let Some(current) = current.upgrade() {
                return current;
            }
        }
        unreachable!("Current node must exists");
    }
}
