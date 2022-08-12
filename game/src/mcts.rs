use std::{rc::{ Weak, Rc }, borrow::Borrow};
use core::cell::RefCell;

#[allow(dead_code)]
pub struct TreeNode {
    parent: Option<Weak<RefCell<TreeNode>>>, // None if this is a root node
    children: Vec<Rc<RefCell<TreeNode>>>,
    current: Option<Weak<RefCell<TreeNode>>>,
    visit_times : u32,  // number of visited times
    prior_probability : f32,  // prior probability from policy network
    q : f32, // action value
    u : f32, // visit-count-adjusted prior score 
}


#[allow(dead_code)]
impl TreeNode {
    pub fn new() -> Rc<RefCell<TreeNode>>
    {
        TreeNode::new_with(|_| () )
    }

    pub fn new_with<F>(mut init : F) -> Rc<RefCell<TreeNode>> where F : FnMut(&mut TreeNode)
    {
        let mut node = TreeNode { 
            parent: None, 
            children: Vec::new(),
            current: None,
            visit_times: 0,
            prior_probability: 0f32,
            q : 0f32,
            u : 0f32,
        };
        init(&mut node);
        let instance = Rc::new(RefCell::new(node));
        instance.borrow_mut().current = Some(Rc::downgrade(&instance));
        instance
    }

    // Create a new child
    pub fn create_child<F>(self: &mut TreeNode, init : F) -> Rc<RefCell<TreeNode>> where F : FnMut(&mut TreeNode) 
    {
        let instance = TreeNode::new_with(init);
        self.append_child(instance.clone());
        instance
    }

    // Append a child
    pub fn append_child(self: &mut TreeNode, child: Rc<RefCell<TreeNode>>) {
        assert!(self.current.is_some());
        // link child node to this one
        child.borrow_mut().parent = Some(self.current.as_ref().unwrap().clone());
        self.children.push(child);
    }

    // Remove a child specified by index
    pub fn remove_child_at(self: &mut TreeNode, index : usize) -> Rc<RefCell<TreeNode>> {
        assert!(index < self.children.len());
        self.children.remove(index)
    }

    // Determine if this is a leaf node
    pub fn is_leaf(self : &TreeNode) -> bool {
        self.children.len() == 0
    }

    // Determine if this is root node
    pub fn is_root(self : &TreeNode) -> bool {
        self.parent.is_none()
    }

    // Update node values from leaf evaluation.
    // leaf_value: the value of subtree evaluation from the current player's perspective.
    pub fn update(self: &mut TreeNode, leaf_value : f32) {
        self.visit_times += 1;
        // Q : a running average of values for all visits.
        self.q += 1.0*(leaf_value - self.q) / (self.visit_times as f32);
    }

    // Like a call to update(), but applied recursively for all ancestors.
    pub fn update_recursive(self: &mut TreeNode, leaf_value : f32) {
        // If it is not root, this node's parent should be updated first.
        if let Some(parent) = &self.parent {
            let parent = parent.upgrade().expect("Unable to upgrade weak reference of parent node");
            parent.borrow_mut().update_recursive(-leaf_value);
        }
        self.update(leaf_value);
    }

    // Calculate and return the value for this node.
    // It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
    // c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
    pub fn compute_value(self : &mut TreeNode, c_puct : f32) -> f32 {
        match &self.parent {
            Some(parent) => {
                let parent = parent.upgrade().expect("Unable to upgrade weak reference of parent node");
                self.u = c_puct * self.prior_probability * f32::sqrt(parent.as_ref().borrow().visit_times as f32) / (1f32 + self.visit_times as f32);
            },
            None => {
                self.u = 0f32;
            }
        }
        return self.q + self.u
    }
}


#[test]
fn test_tree()
{
    let root = TreeNode::new();

    let child1 = TreeNode::new();
    
    root.borrow_mut().append_child(child1.clone());

    let child2 = TreeNode::new_with(|node| {
        node.prior_probability = 1f32;
        node.q = 2f32;
        node.u = 3f32;
    });

    child1.borrow_mut().append_child(child2);

    child1.borrow_mut().remove_child_at(0);

    let child3 = TreeNode::new();
    child1.borrow_mut().append_child(child3);

}