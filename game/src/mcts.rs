use core::cell::RefCell;
use std::{
    collections::HashMap,
    rc::{Rc, Weak},
};

#[allow(dead_code)]
pub struct TreeNode {
    parent: Option<Weak<RefCell<TreeNode>>>, // None if this is a root node
    children: HashMap<(usize /*row*/, usize /*col*/), Rc<RefCell<TreeNode>>>,
    current: Option<Weak<RefCell<TreeNode>>>,
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

#[allow(dead_code)]
impl TreeNode {
    pub fn new(probability: f32) -> Rc<RefCell<TreeNode>> {
        let mut node = TreeNode {
            parent: None,
            children: HashMap::new(),
            current: None,
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
        let instance = TreeNode::new(probability);
        self.append_child(pos, instance.clone());
        instance
    }

    // Append a child
    fn append_child(self: &mut TreeNode, pos: (usize, usize), child: Rc<RefCell<TreeNode>>) {
        assert!(self.current.is_some());
        // link child node to this one
        child.borrow_mut().parent = Some(self.current.as_ref().unwrap().clone());
        self.children.insert(pos, child);
    }

    // Determine if this is a leaf node
    pub fn is_leaf(self: &TreeNode) -> bool {
        self.children.len() == 0
    }

    // Determine if this is root node
    pub fn is_root(self: &TreeNode) -> bool {
        self.parent.is_none()
    }

    // Update node values from leaf evaluation.
    // leaf_value: the value of subtree evaluation from the current player's perspective.
    pub fn update(self: &mut TreeNode, leaf_value: f32) {
        self.visit_times += 1;
        // q is the avarage value in visit_times, initially q is zero
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
    fn compute_score(self: &Self, c_puct: f32) -> f32 {
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
}

// Monte Carlo tree search
pub struct TreeSearcher<F>
where
    F: Fn(Tensor<f32>) -> (Tensor<f32>, f32),
{
    c_puct: f32,
    root: Rc<RefCell<TreeNode>>,
    predict_fn: F,
}

use tensorflow::Tensor;

use crate::game::{RenjuBoard, TerminalState};
impl<F> TreeSearcher<F>
where
    F: Fn(Tensor<f32> /*(1,4,15,15)*/) -> (Tensor<f32> /*(1,15*15)*/, f32),
{
    pub fn new(c_puct: f32, predict_fn: F) -> Self {
        Self {
            c_puct: c_puct,
            root: TreeNode::new(1f32),
            predict_fn: predict_fn,
        }
    }

    pub fn rollout(self: &mut Self, mut board: RenjuBoard) {
        let mut node = self.root.clone();
        let mut state = TerminalState::default();
        while !state.is_over() {
            // Greedily select next move.
            let child = match node.borrow().select(self.c_puct) {
                None => break, // leaf
                Some((pos, c)) => {
                    state = board.do_move(pos);
                    c
                }
            };
            node = child;
        }

        let mut leaf_value = 0f32;

        match state {
            TerminalState::AvailableMoves(next_moves) => {
                assert!(!next_moves.is_empty());

                // Evaluate the leaf using a network
                let state_tensor = board.get_state_tensor();
                let (log_action_tensor, value) = (self.predict_fn)(state_tensor);
                leaf_value = value;

                // extend children
                for (row, col) in next_moves {
                    let probability = log_action_tensor[row * board.width() + col].exp();
                    node.borrow_mut().create_child((row, col), probability);
                }
            }
            TerminalState::Draw => {
                board.print();
                leaf_value = 0f32;
            }
            TerminalState::BlackWon => {
                board.print();
                if board.is_black_turn() {
                    leaf_value = 1f32;
                } else {
                    leaf_value = -1f32;
                }
            }
            TerminalState::WhiteWon => {
                board.print();
                if board.is_black_turn() {
                    leaf_value = -1f32;
                } else {
                    leaf_value = 1f32;
                }
            }
        };

        node.borrow_mut().update_recursive(-leaf_value);
    }
}
