use crate::game::{RenjuBoard, SquareMatrix, StateTensor, TerminalState};
use crate::model::RenjuModel;
use atomicbox::AtomicBox;
use core::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak as ArcWeak};
use std::{collections::HashMap, rc::Rc};
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::oneshot::{self, Receiver, Sender};

#[derive(Debug)]
pub struct ThreadSafeTreeNode {
    stones: u32, // number of stones on board without couting this node
    action: Option<(usize, usize)>,
    parent: Option<ArcWeak<AtomicBox<Receiver<Self>>>>,
    current: ArcWeak<AtomicBox<Receiver<Self>>>,
    children: HashMap<(usize /*row*/, usize /*col*/), Arc<AtomicBox<Receiver<Self>>>>,
    visit_times: u32, // number of visited times
    probability: f32, // prior probability from policy network

    // UCB is based on the principle of “optimism in the fact of uncertainty”,
    // which basically means if you don’t know which action is best
    // then choose the one that currently looks to be the best.
    // UCB = q + u

    // The first half of the equation will do exactly that:
    // the action that currently has the highest estimated reward will be the chosen action.
    q: f32, // the exploitation part of the equation. average of evaluations of all leaves

    // WATCH THE UNOBSERVED: A SIMPLE APPROACH TO PARALLELIZING MONTE CARLO TREE SEARCH
    // the number of rollouts that have been initiated but not yet completed, which we name as unobserved samples.
    unobserved_times: u32,
}

impl ThreadSafeTreeNode {
    fn new(prob: f32) -> Arc<AtomicBox<Receiver<Self>>> {
        let (tx, rx) = oneshot::channel();

        let node = Arc::new(AtomicBox::new(Box::new(rx)));

        let mut child = ThreadSafeTreeNode {
            stones: 0,
            action: None,
            parent: None,
            current: Arc::downgrade(&node),
            children: HashMap::new(),
            unobserved_times: 0,
            visit_times: 0,
            probability: prob,
            q: 0f32,
        };
        tx.send(child).unwrap();
        node
    }

    // create a child
    fn create_child<F>(
        self: &mut Self,
        pos: (usize, usize),
        mut init: F,
    ) -> Arc<AtomicBox<Receiver<Self>>>
    where
        F: FnMut(&mut ThreadSafeTreeNode),
    {
        let (tx, rx) = oneshot::channel();

        let node = Arc::new(AtomicBox::new(Box::new(rx)));

        let mut child = ThreadSafeTreeNode {
            stones: self.stones + 1,
            action: Some(pos),
            parent: Some(self.current.clone()), // link to this one
            current: Arc::downgrade(&node),
            children: HashMap::new(),
            unobserved_times: 0,
            visit_times: 0,
            probability: 0f32,
            q: 0f32,
        };
        init(&mut child);

        self.children.insert(pos, node.clone());
        tx.send(child).unwrap();
        node
    }

    // recursively access from specific node to its all ancestors till root
    async fn recursive_access<F>(self: &mut Self, mut cb: F) -> Result<(), RecvError>
    where
        F: FnMut(&mut Self) + Send,
    {
        cb(self);
        let mut parent_box = self.parent.as_ref().and_then(|x| x.upgrade());

        while let Some(parent) = parent_box {
            let (tx, rx) = oneshot::channel();
            let prev_rx = parent.swap(Box::new(rx), Ordering::Relaxed);
            let mut parent_node = prev_rx.await?;
            cb(&mut parent_node);
            parent_box = parent_node.parent.as_ref().and_then(|x| x.upgrade());
            tx.send(parent_node).unwrap();
        }

        Ok(())
    }

    pub async fn back_propagate(self: &mut Self, leaf_value: f32) -> Result<(), RecvError> {
        self.recursive_access(|node| {
            node.unobserved_times -= 1;
            node.visit_times += 1;

            // q is the avarage score(evaluabl) in visit_times, initially q is zero
            // q = sum(value) / n
            // Progressive update : new_q = (value - old_q) / n + old_q
            node.q += 1.0 * (leaf_value - node.q) / (node.visit_times as f32);
        })
        .await
    }

    // Calculate and return the value for this node.
    // It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
    // c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
    fn compute_score(self: &Self, c_puct: f32, parent_visit_times: u32) -> f32 {
        // The second half of the equation adds exploration,
        // with the degree of exploration being controlled by the hyper-parameter ‘c’.
        // Effectively this part of the equation provides a measure of the uncertainty for the action’s reward estimate.
        // u = visit-count-adjusted prior score
        let u = c_puct
            * self.probability
            * f32::sqrt(parent_visit_times as f32 + self.unobserved_times as f32)
            / (1f32 + self.visit_times as f32 + self.unobserved_times as f32);
        return self.q + u;
    }

    /// Select a direct child with max UCB(Q+U)
    async fn select(self: &Self, c_puct: f32) -> Result<Option<(Sender<Self>, Self)>, RecvError> {
        let mut selected: Option<(Sender<Self>, _)> = None;
        let mut max_score = f32::MIN;

        // order is important here. be careful to avoid dead lock
        for (_, child) in &self.children {
            let (tx, rx) = oneshot::channel();
            let prev_rx = child.swap(Box::new(rx), Ordering::Relaxed);
            let child_node = prev_rx.await?;

            let score = child_node.compute_score(c_puct, self.visit_times);
            if score > max_score {
                max_score = score;

                if let Some((existing_tx, existing_node)) = selected.take() {
                    existing_tx.send(existing_node).unwrap(); // release non-max one
                }
                selected = Some((tx, child_node));
            } else {
                tx.send(child_node).unwrap(); // release non-max one
            }
        }

        Ok(selected)
    }

    async fn greedy_select_leaf(
        self_ref: &Arc<AtomicBox<Receiver<Self>>>,
        c_puct: f32,
    ) -> Result<(Sender<Self>, Self, Vec<(usize, usize)>), RecvError> {
        let (mut sender, rx) = oneshot::channel();
        let prev_rx = self_ref.swap(Box::new(rx), Ordering::Relaxed);

        let mut moves = Vec::with_capacity(20);

        let mut current_node = prev_rx.await?;
        current_node.unobserved_times += 1;
        loop {
            match current_node.select(c_puct).await? {
                Some((tx, child_node)) => {
                    sender.send(current_node).unwrap();
                    sender = tx;
                    current_node = child_node;
                    current_node.unobserved_times += 1;
                    moves.push(current_node.action.unwrap());
                }
                None => {
                    break;
                }
            };
        }

        Ok((sender, current_node, moves))
    }

    async fn enumerate_children<F>(
        self_ref: &Arc<AtomicBox<Receiver<Self>>>,
        mut cb: F,
    ) -> Result<(), RecvError>
    where
        F: FnMut(&(usize, usize), &Self),
    {
        let (sender, rx) = oneshot::channel();
        let prev_rx = self_ref.swap(Box::new(rx), Ordering::Relaxed);

        let current_node = prev_rx.await?;

        // order is important here. be careful to avoid dead lock
        for (pos, child) in &current_node.children {
            let (tx, rx) = oneshot::channel();
            let prev_rx = child.swap(Box::new(rx), Ordering::Relaxed);
            let child_node = prev_rx.await?;
            cb(pos, &child_node);
            tx.send(child_node).unwrap();
        }
        sender.send(current_node).unwrap();

        Ok(())
    }
}

// Monte Carlo tree search
pub struct MonteCarloTree<M>
where
    M: RenjuModel,
{
    model: M,
    c_puct: f32,
    root: Arc<AtomicBox<Receiver<ThreadSafeTreeNode>>>,
}

impl<M> MonteCarloTree<M>
where
    M: RenjuModel,
{
    pub fn new(c_puct: f32, model: M) -> Self {
        Self {
            c_puct: c_puct,
            root: ThreadSafeTreeNode::new(1f32),
            model: model,
        }
    }

    pub async fn rollout(
        self: &mut Self,
        mut board: RenjuBoard,
        choices: &Vec<(usize, usize)>,
    ) -> Result<(), RecvError> {
        let (sender, mut node, moves) =
            ThreadSafeTreeNode::greedy_select_leaf(&self.root, self.c_puct).await?;

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
                let state_tensor: [StateTensor; 1] = [board.get_state_tensor()];

                let (prob_matrix, score) = self
                    .model
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
                    node.create_child((row, col), |child| {
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

        node.back_propagate(evaluation_score).await?;

        sender.send(node).unwrap();
        Ok(())
    }

    pub async fn get_visit_times(self: &Self) -> Result<SquareMatrix<u32>, RecvError> {
        let mut matrix = SquareMatrix::default();
        // get visit times of direct children of root
        ThreadSafeTreeNode::enumerate_children(&self.root, |(row, col), child| {
            matrix[*row][*col] = child.visit_times;
        })
        .await?;
        Ok(matrix)
    }

    // temperature parameter in (0, 1] controls the level of exploration
    pub async fn get_move_probability(
        self: &mut Self,
        temperature: f32,
    ) -> Result<Vec<((usize, usize) /*pos*/, f32 /* probability */)>, RecvError> {
        let mut pairs = Vec::with_capacity(50);
        let mut max_log_visit_times = 0f32;
        // calc the move probabilities based on visit counts in top level
        ThreadSafeTreeNode::enumerate_children(&self.root, |pos, child| {
            let log_visit_times =
                1f32 / temperature * (1e-10 /*avoid zero*/ + child.visit_times as f32).ln();
            if log_visit_times > max_log_visit_times {
                max_log_visit_times = log_visit_times;
            }
            pairs.push((*pos, log_visit_times));
        })
        .await?;

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

    pub async fn update_with_position(
        self: &mut Self,
        pos: (usize, usize),
    ) -> Result<(), RecvError> {
        let (_, rx) = oneshot::channel();
        let prev_rx = self.root.swap(Box::new(rx), Ordering::Relaxed);

        let mut node = prev_rx.await?;

        let child_ref = {
            node.children
                .remove(&pos)
                .or_else(|| Some(node.create_child(pos, |_| {})))
                .unwrap()
        };

        let (tx, rx) = oneshot::channel();
        let prev_rx = child_ref.swap(Box::new(rx), Ordering::Relaxed);

        let mut child = prev_rx.await?;
        child.parent = None;
        tx.send(child).unwrap();

        self.root = child_ref;

        Ok(())
    }
}
