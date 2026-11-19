//! Linear scan register allocation from IR variables to the 256 VM
//! registers.

use crate::compiler::ir::block::ControlFlowGraph;
use crate::compiler::ir::inst::IrInst;
use crate::compiler::ir::liveness::LivenessInfo;
use crate::compiler::ir::types::IrVar;
use crate::vm::hash::{FxHashMap, new_map, with_capacity};

/// Fixed-capacity 256-bit register bitset backed by four `u64` words.
/// Avoids heap allocations and finds the lowest free register in O(1).
#[derive(Clone, Copy, Default)]
struct RegSet([u64; 4]);

impl RegSet {
    fn from_range(start: u8, end_exclusive: u8) -> Self {
        let mut set = Self::default();
        for reg in start..end_exclusive {
            set.insert(reg);
        }
        set
    }

    #[inline]
    fn insert(&mut self, reg: u8) {
        self.0[(reg >> 6) as usize] |= 1u64 << (reg & 63);
    }

    #[inline]
    fn remove(&mut self, reg: u8) {
        self.0[(reg >> 6) as usize] &= !(1u64 << (reg & 63));
    }

    #[inline]
    fn contains(&self, reg: u8) -> bool {
        (self.0[(reg >> 6) as usize] & (1u64 << (reg & 63))) != 0
    }

    #[inline]
    fn lowest(&self) -> Option<u8> {
        for (idx, &word) in self.0.iter().enumerate() {
            if word != 0 {
                return Some(((idx as u8) << 6) | (word.trailing_zeros() as u8));
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct RegisterAllocation {
    pub mapping: FxHashMap<IrVar, u8>,
    pub max_registers: u8,
}

impl RegisterAllocation {
    pub fn get(&self, var: IrVar) -> Option<u8> {
        self.mapping.get(&var).copied()
    }
}

pub fn allocate_registers(
    cfg: &ControlFlowGraph,
    num_params: u8,
) -> Result<RegisterAllocation, String> {
    let liveness = LivenessInfo::compute(cfg);
    let intervals_map = liveness.compute_live_intervals(cfg);

    let mut intervals: Vec<_> = intervals_map.into_values().collect();

    // A closure holds the register of each variable it captures, so that
    // register must not be handed to anything else once the variable's last
    // read goes by: the closure would see whatever overwrote it.
    let captured = crate::compiler::ir::opt::captured_vars(cfg);
    if !captured.is_empty() {
        for interval in &mut intervals {
            if captured.contains(&interval.var) {
                interval.end = usize::MAX;
            }
        }
    }

    // Collect copy hints for register coalescing
    let mut copy_hints: FxHashMap<IrVar, Vec<IrVar>> = new_map();
    for block in &cfg.blocks {
        for inst in &block.instructions {
            match inst {
                IrInst::Move { dst, src } if dst != src => {
                    copy_hints.entry(*dst).or_default().push(*src);
                    copy_hints.entry(*src).or_default().push(*dst);
                }
                IrInst::Phi { dst, incoming } => {
                    for (_, src) in incoming {
                        if dst != src {
                            copy_hints.entry(*dst).or_default().push(*src);
                            copy_hints.entry(*src).or_default().push(*dst);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Sort by start position, with the variable id breaking ties so the
    // allocation does not depend on hash order.
    intervals.sort_by_key(|intv| (intv.start, intv.end, intv.var.0));

    let mut mapping: FxHashMap<IrVar, u8> = with_capacity(intervals.len() + num_params as usize);

    // 1. Pre-allocate parameter registers: 0..num_params
    for i in 0..num_params {
        mapping.insert(IrVar(i as u32), i);
    }

    // 2. Linear scan register allocation with coalescing
    let mut free_registers = RegSet::from_range(num_params, 255);
    // Active intervals: (end_point, register, IrVar)
    let mut active: Vec<(usize, u8, IrVar)> = Vec::new();
    let mut max_reg = num_params;

    for interval in intervals {
        // Skip already mapped parameters
        if mapping.contains_key(&interval.var) {
            continue;
        }

        // Expire old intervals
        active.retain(|&(end, reg, _var)| {
            if end < interval.start {
                free_registers.insert(reg);
                false
            } else {
                true
            }
        });

        // Coalescing: if any copy partner of interval.var is in active and ends at interval.start,
        // it dies right at this instruction (the copy). We can retire it early to reuse its register.
        if let Some(hints) = copy_hints.get(&interval.var) {
            for &partner in hints {
                if let Some(pos) = active
                    .iter()
                    .position(|&(end, _, v)| v == partner && end <= interval.start)
                {
                    let (_, reg, _) = active.remove(pos);
                    free_registers.insert(reg);
                }
            }
        }

        // Try allocating a preferred register from copy hints first
        let preferred_reg = copy_hints.get(&interval.var).and_then(|hints| {
            hints.iter().find_map(|partner| {
                mapping
                    .get(partner)
                    .copied()
                    .filter(|&r| free_registers.contains(r))
            })
        });

        let chosen_reg = if let Some(pref) = preferred_reg {
            pref
        } else if let Some(lowest) = free_registers.lowest() {
            lowest
        } else {
            return Err(format!(
                "register allocation failure: register limit (255) exceeded for variable {:?}",
                interval.var
            ));
        };

        free_registers.remove(chosen_reg);
        mapping.insert(interval.var, chosen_reg);
        active.push((interval.end, chosen_reg, interval.var));
        if chosen_reg.saturating_add(1) > max_reg {
            max_reg = chosen_reg.saturating_add(1);
        }
    }

    Ok(RegisterAllocation {
        mapping,
        max_registers: max_reg,
    })
}

#[cfg(test)]
mod tests {
    use super::RegSet;

    #[test]
    fn test_regset_word_boundaries_and_lowest() {
        let mut set = RegSet::from_range(2, 255);
        assert!(!set.contains(0));
        assert!(!set.contains(1));
        assert!(set.contains(2));
        assert!(set.contains(63));
        assert!(set.contains(64));
        assert!(set.contains(127));
        assert!(set.contains(128));
        assert!(set.contains(191));
        assert!(set.contains(192));
        assert!(set.contains(254));
        assert!(!set.contains(255));

        for expected in 2..255u8 {
            assert_eq!(set.lowest(), Some(expected));
            set.remove(expected);
            assert!(!set.contains(expected));
        }
        assert_eq!(set.lowest(), None);

        set.insert(192);
        set.insert(64);
        assert_eq!(set.lowest(), Some(64));
        set.remove(64);
        assert_eq!(set.lowest(), Some(192));
        set.remove(192);
        assert_eq!(set.lowest(), None);
    }
}
