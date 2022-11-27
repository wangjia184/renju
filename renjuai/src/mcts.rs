/*
 * (C) Copyright 2022 Jerry.Wang (https://github.com/wangjia184).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::game::{RenjuBoard, SquareMatrix, StateTensor, TerminalState};
use crossbeam::atomic::AtomicCell;
use std::collections::HashMap;

use async_lock::RwLock;
use async_oneshot::*;
use std::sync::{Arc, Weak as ArcWeak};
// thread-safe tree node
pub struct TreeNode {
    stones: u32, // number of stones on board without couting this node
    action: Option<(usize, usize)>,
    parent: Option<ArcWeak<AtomicCell<Receiver<Self>>>>,
    current: ArcWeak<AtomicCell<Receiver<Self>>>,
    children: HashMap<(usize /*row*/, usize /*col*/), Arc<AtomicCell<Receiver<Self>>>>,
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

// opened tree node for read or update
struct OpenedTreeNode(Option<TreeNode>, Option<Sender<TreeNode>>);
impl Drop for OpenedTreeNode {
    fn drop(&mut self) {
        assert!(self.0.is_some());
        assert!(self.1.is_some());

        if let Err(_) = self.1.take().unwrap().send(self.0.take().unwrap()) {
            panic!("Unable to send");
        }
    }
}
impl OpenedTreeNode {
    fn get_mut(&mut self) -> &mut TreeNode {
        if let Some(x) = &mut self.0 {
            x
        } else {
            unreachable!()
        }
    }

    fn get(&self) -> &TreeNode {
        if let Some(x) = &self.0 {
            x
        } else {
            unreachable!()
        }
    }
}

// children nodes for selection
// to avoid occupying the parent node when selecting from its children
struct TreeNodeChildren {
    pairs: Vec<(
        (usize /*row*/, usize /*col*/),
        Arc<AtomicCell<Receiver<TreeNode>>>,
    )>,
    parent_visit_times: u32,
}

impl TreeNodeChildren {
    /// Select a direct child with max UCB(Q+U)
    async fn select(self: &Self, c_puct: f32) -> Result<OpenedTreeNode, Closed> {
        let mut selected: Option<OpenedTreeNode> = None;
        let mut max_score = f32::MIN;

        // order is important here. be careful to avoid dead lock
        for (_, child) in &self.pairs {
            let child_node = TreeNode::open_node(child).await?;

            let score = child_node
                .get()
                .compute_score(c_puct, self.parent_visit_times);
            if score > max_score {
                max_score = score;
                selected = Some(child_node);
            }
        }
        assert!(selected.is_some());
        Ok(selected.unwrap())
    }
}

impl TreeNode {
    fn new(prob: f32, action: Option<(usize, usize)>) -> Arc<AtomicCell<Receiver<Self>>> {
        let (mut tx, rx) = oneshot();

        let node = Arc::new(AtomicCell::new(rx));

        let child = TreeNode {
            stones: 0,
            action: action,
            parent: None,
            current: Arc::downgrade(&node),
            children: HashMap::new(),
            visit_times: 0,
            probability: prob,
            q: 0f32,
        };
        if let Err(_) = tx.send(child) {
            panic!("tx.send(child) failed");
        }
        node
    }

    // create a child
    fn create_child<F>(
        self: &mut Self,
        pos: (usize, usize),
        mut init: F,
    ) -> Arc<AtomicCell<Receiver<Self>>>
    where
        F: FnMut(&mut TreeNode),
    {
        let (mut tx, rx) = oneshot();

        let node = Arc::new(AtomicCell::new(rx));

        let mut child = TreeNode {
            stones: self.stones + 1,
            action: Some(pos),
            parent: Some(self.current.clone()), // link to this one
            current: Arc::downgrade(&node),
            children: HashMap::new(),
            visit_times: 0,
            probability: 0f32,
            q: 0f32,
        };
        init(&mut child);

        self.children.insert(pos, node.clone());
        if let Err(_) = tx.send(child) {
            panic!("tx.send(child) failed");
        }
        node
    }

    async fn open_node(
        reference: &Arc<AtomicCell<Receiver<Self>>>,
    ) -> Result<OpenedTreeNode, Closed> {
        let (tx, rx) = oneshot();
        let prev_rx = reference.swap(rx);
        let node = prev_rx.await?;

        Ok(OpenedTreeNode(Some(node), Some(tx)))
    }

    // recursively access from specific node to its all ancestors till root
    async fn back_propagate(
        mut reference: Arc<AtomicCell<Receiver<Self>>>,
        mut leaf_value: f32,
    ) -> Result<(), Closed> {
        loop {
            leaf_value *= -1f32;
            let mut current_node = Self::open_node(&reference).await?;
            current_node.get_mut().update(leaf_value);
            if let Some(x) = current_node.get().parent.as_ref().and_then(|x| x.upgrade()) {
                reference = x;
            } else {
                break;
            }
        }

        Ok(())
    }

    pub fn update(self: &mut Self, leaf_value: f32) {
        self.visit_times += 1;

        // q is the avarage score(evaluabl) in visit_times, initially q is zero
        // q = sum(value) / n
        // Progressive update : new_q = (value - old_q) / n + old_q
        self.q += 1.0 * (leaf_value - self.q) / (self.visit_times as f32);
    }

    // Calculate and return the value for this node.
    // It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
    // c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
    fn compute_score(self: &Self, c_puct: f32, parent_visit_times: u32) -> f32 {
        // The second half of the equation adds exploration,
        // with the degree of exploration being controlled by the hyper-parameter ‘c’.
        // Effectively this part of the equation provides a measure of the uncertainty for the action’s reward estimate.
        // u = visit-count-adjusted prior score
        let u = c_puct * self.probability * f32::sqrt(parent_visit_times as f32)
            / (1f32 + self.visit_times as f32);
        return self.q + u;
    }

    fn get_children(self: &Self) -> TreeNodeChildren {
        assert!(!self.children.is_empty());
        let mut children = TreeNodeChildren {
            parent_visit_times: self.visit_times,
            pairs: self
                .children
                .iter()
                .map(|(pos, cell)| (*pos, cell.clone()))
                .collect(),
        };
        children.pairs.sort_by_key(|pair| pair.0);
        children
    }

    async fn greedy_select_leaf(
        self_ref: &Arc<AtomicCell<Receiver<Self>>>,
        c_puct: f32,
    ) -> Result<(OpenedTreeNode, Vec<(usize, usize)>), Closed> {
        let mut moves = Vec::with_capacity(20);

        let mut current_node = Self::open_node(self_ref).await?;

        loop {
            if current_node.get().children.is_empty() {
                break;
            } else {
                let children = current_node.get().get_children();
                drop(current_node); //release parent node before select children
                current_node = children.select(c_puct).await?;

                moves.push(current_node.get().action.unwrap());
            }
        }

        Ok((current_node, moves))
    }

    async fn enumerate_children<F>(
        self_ref: &Arc<AtomicCell<Receiver<Self>>>,
        mut cb: F,
    ) -> Result<(), Closed>
    where
        F: FnMut(&(usize, usize), &Self),
    {
        let current_node = Self::open_node(self_ref).await?;

        // order is important here. be careful to avoid dead lock
        for (pos, child) in &current_node.get().children {
            let child_node = Self::open_node(child).await?;

            cb(pos, child_node.get());
        }

        Ok(())
    }
}

// Monte Carlo tree search
pub struct MonteCarloTree {
    c_puct: f32,
    root: RwLock<Arc<AtomicCell<Receiver<TreeNode>>>>,
}

impl MonteCarloTree {
    pub fn new(c_puct: f32) -> Self {
        Self {
            c_puct: c_puct,
            root: RwLock::new(TreeNode::new(1f32, None)),
        }
    }

    pub fn new_with_position(pos: (usize, usize), c_puct: f32) -> Self {
        Self {
            c_puct: c_puct,
            root: RwLock::new(TreeNode::new(1f32, Some(pos))),
        }
    }

    pub async fn rollout<F>(
        self: &Self,
        mut board: RenjuBoard,
        choices: &Vec<(usize, usize)>,
        mut predict_fn: F,
    ) -> Result<(), Closed>
    where
        F: FnMut(StateTensor) -> (SquareMatrix, f32),
    {
        let root = self.root.read().await.clone();

        let (mut node, moves) = TreeNode::greedy_select_leaf(&root, self.c_puct).await?;

        //assert_eq!(board.get_last_move(), root.action);

        let mut state: Option<TerminalState> = None;
        if !moves.is_empty() {
            moves.into_iter().for_each(|pos| {
                state = Some(board.do_move(pos));
            });
        } else {
            if choices.len() > 1 {
                //println!("Never explored before");
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
                let (prob_matrix, score) = predict_fn(board.get_state_tensor());

                // black and white are placed in turns
                // if `node` is a black move, then `score` is an evaluation from white's perspective.
                // Since this is a zero-sum game. Positive score means advantages for white, then disadvantages for black.
                // That's why we need negative score because it is from opposite perspective.
                evaluation_score = -score;

                // extend children
                for (row, col) in choices {
                    let probability = prob_matrix[row][col];
                    node.get_mut().create_child((row, col), |child| {
                        child.probability = probability;
                    });
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

        node.get_mut().update(evaluation_score);
        if let Some(parent) = node.get().parent.as_ref().and_then(|x| x.upgrade()) {
            drop(node);
            TreeNode::back_propagate(parent, evaluation_score).await?;
        }

        Ok(())
    }

    pub async fn get_visit_times(self: &Self) -> Result<SquareMatrix<u32>, Closed> {
        let mut matrix = SquareMatrix::default();
        // get visit times of direct children of root
        let root = self.root.read().await.clone();
        TreeNode::enumerate_children(&root, |(row, col), child| {
            matrix[*row][*col] = child.visit_times;
        })
        .await?;
        Ok(matrix)
    }

    // temperature parameter in (0, 1] controls the level of exploration
    pub async fn get_move_probability(
        self: &Self,
        temperature: f32,
    ) -> Result<Vec<((usize, usize) /*pos*/, f32 /* probability */)>, Closed> {
        let mut pairs = Vec::with_capacity(50);
        let mut max_log_visit_times = 0f32;
        let mut total_visit_times = 0;
        // calc the move probabilities based on visit counts in top level
        let root = self.root.read().await.clone();
        TreeNode::enumerate_children(&root, |pos, child| {
            let log_visit_times =
                1f32 / temperature * (1e-10 /*avoid zero*/ + child.visit_times as f32).ln();
            if log_visit_times > max_log_visit_times {
                max_log_visit_times = log_visit_times;
            }
            total_visit_times += child.visit_times;
            pairs.push((*pos, log_visit_times));
        })
        .await?;

        assert_ne!(total_visit_times, 0);

        // softmax
        let mut sum = 0f32;
        for (_, log_visit_times) in &mut pairs {
            *log_visit_times = (*log_visit_times - max_log_visit_times).exp();
            sum += *log_visit_times;
        }

        for (_, prob) in &mut pairs {
            *prob /= sum;
        }

        Ok(pairs)
    }

    pub async fn update_with_position(self: &Self, pos: (usize, usize)) -> Result<(), Closed> {
        let root = self.root.read().await.clone();

        let mut node = TreeNode::open_node(&root).await?;

        let child_ref = {
            node.get_mut()
                .children
                .remove(&pos)
                .or_else(|| Some(node.get_mut().create_child(pos, |_| {})))
                .unwrap()
        };

        let mut child = TreeNode::open_node(&child_ref).await?;
        child.get_mut().parent = None;

        *self.root.write().await = child_ref;

        Ok(())
    }
}
