//! Dominator trees and dominance frontiers.

use crate::compiler::ir::block::ControlFlowGraph;
use crate::compiler::ir::types::IrLabel;
use crate::vm::hash::{FxHashMap, FxHashSet, new_map, with_capacity};

/// Immediate dominators without tree children or dominance frontiers.
/// Passes that only query dominance (e.g. LICM) build this lighter type so a
/// `DominatorTree` is never left half-populated.
#[derive(Debug, Clone)]
pub struct ImmediateDominators {
    pub idom: FxHashMap<IrLabel, IrLabel>,
}

impl ImmediateDominators {
    pub fn build(cfg: &ControlFlowGraph) -> Self {
        let entry = cfg.entry_label;
        let mut idom: FxHashMap<IrLabel, IrLabel> = with_capacity(cfg.blocks.len());
        let mut label_to_idx: FxHashMap<IrLabel, usize> = with_capacity(cfg.blocks.len());

        for (idx, b) in cfg.blocks.iter().enumerate() {
            label_to_idx.insert(b.label, idx);
        }

        idom.insert(entry, entry);

        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for block in &cfg.blocks {
                let b = block.label;
                if b == entry {
                    continue;
                }

                // Find first processed predecessor
                let mut processed_pred = None;
                for p in &block.predecessors {
                    if idom.contains_key(p) {
                        processed_pred = Some(*p);
                        break;
                    }
                }

                let Some(mut new_idom) = processed_pred else {
                    continue;
                };

                for p in &block.predecessors {
                    let pred = *p;
                    if pred != new_idom && idom.contains_key(&pred) {
                        new_idom = intersect(pred, new_idom, &idom, &label_to_idx);
                    }
                }

                if idom.get(&b) != Some(&new_idom) {
                    idom.insert(b, new_idom);
                    changed = true;
                }
            }
        }

        Self { idom }
    }

    pub fn dominates(&self, a: IrLabel, b: IrLabel) -> bool {
        idom_dominates(&self.idom, a, b)
    }
}

fn idom_dominates(idom: &FxHashMap<IrLabel, IrLabel>, a: IrLabel, mut b: IrLabel) -> bool {
    if a == b {
        return true;
    }
    let mut steps = 0;
    while let Some(&parent) = idom.get(&b) {
        if parent == a {
            return true;
        }
        if parent == b {
            break;
        }
        b = parent;
        steps += 1;
        if steps > 1000 {
            break;
        }
    }
    false
}

#[derive(Debug, Clone)]
pub struct DominatorTree {
    pub idom: FxHashMap<IrLabel, IrLabel>,
    pub children: FxHashMap<IrLabel, Vec<IrLabel>>,
    pub frontiers: FxHashMap<IrLabel, FxHashSet<IrLabel>>,
}

impl DominatorTree {
    pub fn build(cfg: &ControlFlowGraph) -> Self {
        let idom = ImmediateDominators::build(cfg).idom;

        // Build tree children
        let mut children: FxHashMap<IrLabel, Vec<IrLabel>> = new_map();
        for (&node, &parent) in &idom {
            if node != parent {
                children.entry(parent).or_default().push(node);
            }
        }

        // Build dominance frontiers
        let mut frontiers: FxHashMap<IrLabel, FxHashSet<IrLabel>> = with_capacity(cfg.blocks.len());
        for b in &cfg.blocks {
            frontiers.entry(b.label).or_default();
        }

        for block in &cfg.blocks {
            let b = block.label;
            if block.predecessors.len() >= 2 {
                for p in &block.predecessors {
                    let mut runner = *p;
                    let target_idom = idom.get(&b).copied().unwrap_or(b);

                    let mut steps = 0;
                    while runner != target_idom && steps < 500 {
                        frontiers.entry(runner).or_default().insert(b);
                        if let Some(&next_runner) = idom.get(&runner) {
                            if next_runner == runner {
                                break;
                            }
                            runner = next_runner;
                        } else {
                            break;
                        }
                        steps += 1;
                    }
                }
            }
        }

        Self {
            idom,
            children,
            frontiers,
        }
    }

    pub fn dominates(&self, a: IrLabel, b: IrLabel) -> bool {
        idom_dominates(&self.idom, a, b)
    }
}

fn intersect(
    mut b1: IrLabel,
    mut b2: IrLabel,
    idom: &FxHashMap<IrLabel, IrLabel>,
    label_to_idx: &FxHashMap<IrLabel, usize>,
) -> IrLabel {
    let mut steps = 0;
    while b1 != b2 && steps < 1000 {
        steps += 1;
        let mut idx1 = label_to_idx.get(&b1).copied().unwrap_or(0);
        let mut idx2 = label_to_idx.get(&b2).copied().unwrap_or(0);

        while idx1 > idx2 {
            if let Some(&p) = idom.get(&b1) {
                if p == b1 {
                    break;
                }
                b1 = p;
                idx1 = label_to_idx.get(&b1).copied().unwrap_or(0);
            } else {
                break;
            }
        }
        while idx2 > idx1 {
            if let Some(&p) = idom.get(&b2) {
                if p == b2 {
                    break;
                }
                b2 = p;
                idx2 = label_to_idx.get(&b2).copied().unwrap_or(0);
            } else {
                break;
            }
        }
    }
    b1
}
