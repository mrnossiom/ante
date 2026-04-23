//! Inline [super::Instruction::Handle]s whose handler cases all use `resume` in tail position
//! into plain MIR control flow using closures. The transformation resembles function inlining:
//! the body closure is pasted in, each [super::Instruction::Perform] is replaced with the inlined
//! handler, and `resume v` in the handler becomes a jump to the continuation block with `v` as its
//! argument.
//!
//! If any case uses `resume` outside of a tail-position, the Handle is left for
//! [crate::mir::effect_lowering] to lower into coroutine primitives.

use crate::mir::Mir;

impl Mir {
    #[allow(dead_code)]
    pub(crate) fn optimize_tail_resume(self) -> Self {
        self
    }
}
